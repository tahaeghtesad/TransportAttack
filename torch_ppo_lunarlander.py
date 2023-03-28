import json
import os
import gymnasium as gym
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from util.torch.math import r2_score
from util.torch.misc import allocate_best_device


class QModel(torch.nn.Module):
    def __init__(self, env, config):
        super(QModel, self).__init__()
        action_shape = env.action_space.sample().shape
        state_shape = env.observation_space.sample().shape

        self.dense_layers = torch.nn.ModuleList([
            torch.nn.Linear(
                config['model_config']['dense_layers'][i - 1]['size'] if i != 0 else state_shape[0],
                config['model_config']['dense_layers'][i]['size']
            )
            for i in range(len(config['model_config']['dense_layers']))
        ])

        self.dense_layers_activation = [
            getattr(torch.nn.functional, config['model_config']['dense_layers'][i]['activation'])
            for i in range(len(config['model_config']['dense_layers']))
        ]

        self.output_layer = torch.nn.Linear(
            config['model_config']['dense_layers'][-1]['size'],
            1
        )

        for l in self.dense_layers:
            torch.nn.init.xavier_uniform_(l.weight)
            torch.nn.init.zeros_(l.bias)

    def forward(self, states):
        x = states
        for a, l in zip(self.dense_layers_activation, self.dense_layers):
            x = l(x)
            x = a(x)
        x = self.output_layer(x)
        return x


class ProbabilisticActorModel(torch.nn.Module):
    def __init__(self, env, config):
        super(ProbabilisticActorModel, self).__init__()
        action_shape = env.action_space.sample().shape
        state_shape = env.observation_space.sample().shape

        self.dense_layers = torch.nn.ModuleList([
            torch.nn.Linear(
                config['model_config']['dense_layers'][i - 1]['size'] if i != 0 else state_shape[0],
                config['model_config']['dense_layers'][i]['size']
            )
            for i in range(len(config['model_config']['dense_layers']))
        ])

        self.dense_layers_activation = [
            getattr(torch.nn.functional, config['model_config']['dense_layers'][i]['activation'])
            for i in range(len(config['model_config']['dense_layers']))
        ]

        self.output_layer = torch.nn.Linear(
            config['model_config']['dense_layers'][-1]['size'],
            2 * action_shape[0]
        )

        for l in self.dense_layers:
            torch.nn.init.xavier_uniform_(l.weight)
            torch.nn.init.zeros_(l.bias)

    def forward(self, states):
        x = states
        for a, l in zip(self.dense_layers_activation, self.dense_layers):
            x = l(x)
            x = a(x)
        x = self.output_layer(x)
        loc, scale = torch.split(x, 2, dim=1)
        return loc, 2 * torch.nn.functional.sigmoid(scale)


class PPOModel(torch.nn.Module):
    def __init__(self, env, config, device):
        super().__init__()
        self.device = device
        self.actor = ProbabilisticActorModel(env, config)
        self.critic = QModel(env, config)

        self.gamma = config['rl_config']['gamma']
        self.lam = config['rl_config']['lam']
        self.clip_param = config['rl_config']['clip_param']
        self.kl_coef = config['rl_config']['kl_coef']
        self.entropy_coef = config['rl_config']['entropy_coef']
        self.reward_clip = config['rl_config']['reward_clip']

        self.optimizer = getattr(torch.optim, config['model_config']['optimizer']['class_name'])(self.parameters(), **
        config['model_config']['optimizer']['config'])
        self.criterion = getattr(torch.nn, config['model_config']['loss'])()

        self.to(self.device)

    def forward(self, states):
        means, stds = self.actor.forward(states)
        distribution = torch.distributions.MultivariateNormal(loc=means, scale_tril=torch.diag_embed(stds))
        actions = distribution.sample()
        return actions

    def get_value(self, states):
        return self.critic.forward(states)

    def get_action_distribution(self, states):
        means, stds = self.actor.forward(states)
        return torch.distributions.MultivariateNormal(loc=means, scale_tril=torch.diag_embed(stds))

    def calculate_gae(self, rewards, state_val, next_state_val, dones, gamma, lam):
        gae = 0
        gae_list = torch.zeros(len(rewards), device=self.device)
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * next_state_val[0] * (1 - dones[i]) - state_val[i]
            gae = delta + gamma * lam * (1 - dones[i]) * gae
            gae_list[i] = gae

        normalized_gae = (gae_list - gae_list.mean()) / (gae_list.std() + 1e-8)
        return normalized_gae

    def update(self, states, actions, next_states, rewards, dones):
        states = torch.tensor(states).float().to(self.device)
        actions = torch.tensor(actions).float().to(self.device)
        next_states = torch.tensor(next_states).float().to(self.device)
        rewards = torch.unsqueeze(torch.tensor(rewards).float(), dim=1).to(self.device)
        rewards = torch.clamp(rewards, -self.reward_clip, self.reward_clip)
        dones = torch.unsqueeze(torch.tensor(dones).float(), dim=1).to(self.device)

        with torch.no_grad():
            old_distributions = self.get_action_distribution(states)
            next_values = self.get_value(next_states)
            target_values = rewards + (1 - dones) * self.gamma * next_values
            gae = self.calculate_gae(rewards, self.get_value(states), next_values, dones, self.gamma, self.lam)

            old_log_probs = old_distributions.log_prob(actions)

        new_distributions = self.get_action_distribution(states)
        r_t = torch.exp(new_distributions.log_prob(actions) - old_log_probs)

        q_values = self.get_value(states)
        l_crit = self.criterion(q_values, target_values)
        l_clip = - torch.mean(
            torch.min(torch.clamp(r_t, 1 - self.clip_param,
                                 1 + self.clip_param) * gae, r_t * gae))
        l_entropy = - torch.mean(new_distributions.entropy())
        l_kl = torch.mean(torch.distributions.kl_divergence(new_distributions, old_distributions))

        l_surrogate = l_crit + l_clip + l_entropy * self.entropy_coef + l_kl * \
                      self.kl_coef

        self.optimizer.zero_grad()
        l_surrogate.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return dict(
            loss=l_crit.cpu().data.numpy(),
            r2=r2_score(target_values, q_values).cpu().data.numpy(),
            max_q=target_values.max().cpu().data.numpy(),
            min_q=target_values.min().cpu().data.numpy(),
            mean_q=target_values.mean().cpu().data.numpy(),
            max_gae=gae.max().cpu().data.numpy(),
            min_gae=gae.min().cpu().data.numpy(),
            mean_gae=gae.mean().cpu().data.numpy(),
            entropy=l_entropy.cpu().data.numpy(),
            gae=gae.cpu().data.numpy(),
        )


if __name__ == '__main__':
    run_id = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    os.makedirs(f'logs/{run_id}/weights')
    writer = SummaryWriter(f'logs/{run_id}/')
    device = allocate_best_device()

    config = dict(
        env='LunarLander-v2',
        env_config=dict(
            continuous=True,
        ),
        model_config=dict(
            optimizer=dict(
                class_name='Adam',
                config=dict(
                    lr=2e-6
                )
            ),
            loss='MSELoss',
            dense_layers=[
                dict(size=512, activation='relu'),
                dict(size=512, activation='relu'),
                dict(size=512, activation='relu'),
            ],
        ),
        rl_config=dict(
            gamma=0.99,
            lam=0.95,
            clip_param=0.2,
            entropy_coef=0.2,
            kl_coef=0.4,
            reward_scale=1.0,
            reward_clip=20.0
        ),
        training_config=dict(
            num_episodes=1_000_000,
            batch_size=64,
            trajectory_len=1024,
        ),
    )

    with open(f'logs/{run_id}/config.json', 'w') as f:
        json.dump(config, f)

    env = gym.wrappers.TimeLimit(gym.make(config['env'], **config['env_config']),
                                 max_episode_steps=config['training_config']['trajectory_len'])

    model = PPOModel(env, config, device)

    current_episode = 0
    global_step = 0
    pbar = tqdm(range(config['training_config']['num_episodes']))

    while current_episode < config['training_config']['num_episodes']:

        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []

        obs = None
        done = True
        truncated = True

        for step in range(config['training_config']['batch_size']):

            if done or truncated:
                obs = env.reset()[0]
                done = False
                truncated = False

            action = model.forward(torch.unsqueeze(torch.from_numpy(obs), dim=0).float().to(device)).cpu().data.numpy()[0]
            next_obs, reward, done, truncated, info = env.step(action)

            states.append(obs)
            actions.append(action)
            next_states.append(next_obs)
            rewards.append(reward)
            dones.append(done)

            obs = next_obs

        states = np.array(states)
        actions = np.array(actions)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        dones = np.array(dones)

        stats = model.update(states, actions, next_states, rewards, dones)
        global_step += 1
        pbar.update(1)
        pbar.set_description(
            f'Loss {stats["loss"]:.4f} | '
            f'R2 {stats["r2"]:.6f} | '
            f'MaxQ {stats["max_q"]:.4f} | '
            f'MeanQ {stats["mean_q"]:.4f} | '
            f'MinQ {stats["min_q"]:.4f} | '
            # f'Episode {episode} | '
            # f'Len {step} | '
            f'CumReward {np.sum(rewards):.4f} | '
            # f'DisReward {discounted_reward:.4f} | '
            # f'Eps {epsilon.get_current_epsilon():.4f} | '
            # f'Noise {noise.get_current_noise():.2f} | '
            # f'ReplayBuffer {buffer.size()}'
        )

        writer.add_scalar('model/loss', stats['loss'], global_step)
        writer.add_scalar('model/r2', max(stats['r2'], -1), global_step)
        writer.add_scalar('model/entropy', stats['entropy'], global_step)

        writer.add_scalar('gae/max_gae', stats['max_gae'], global_step)
        writer.add_scalar('gae/mean_gae', stats['mean_gae'], global_step)
        writer.add_scalar('gae/min_gae', stats['min_gae'], global_step)
        writer.add_histogram('gae/gae', stats['gae'], global_step)

        writer.add_scalar('q/max_q', stats['max_q'], global_step)
        writer.add_scalar('q/mean_q', stats['mean_q'], global_step)
        writer.add_scalar('q/min_q', stats['min_q'], global_step)

        if current_episode % 100 == 0:
            torch.save(model.state_dict(), f'logs/{run_id}/weights/{current_episode}.pt')

            test_done = False
            test_truncated = False
            test_obs = env.reset()[0]
            test_episode_reward = 0
            test_episode_steps = 0
            while not test_done and not test_truncated:
                loc, _ = model.actor.forward(torch.unsqueeze(torch.from_numpy(test_obs), dim=0).float().to(device))
                test_action = loc.cpu().data.numpy()[0]
                test_obs, test_reward, test_done, test_truncated, test_info = env.step(test_action)
                test_episode_reward += test_reward
                test_episode_steps += 1

            writer.add_scalar('test/episode_reward', test_episode_reward, global_step)
            writer.add_scalar('test/episode_steps', test_episode_steps, global_step)

            done = True
            truncated = True