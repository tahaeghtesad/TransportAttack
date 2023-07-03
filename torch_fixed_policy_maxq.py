import json
import os
from datetime import datetime

import gym
import numpy as np
import torch
from torch.utils import tensorboard as tb
from tqdm import tqdm

import util.rl.exploration
from attack_heuristics import GreedyRiderVector, Random
from transport_env.NetworkEnv import TransportationNetworkEnvironment
from util.rl.experience_replay import ExperienceReplay
from util.torch.gcn import GraphConvolutionLayer
from util.torch.math import r2_score
from util.torch.misc import allocate_best_device


class DQNModel(torch.nn.Module):
    def __init__(self, env, config):
        super(DQNModel, self).__init__()
        action_shape = env.action_space.sample().shape  # (76, )
        state_shape = env.observation_space.sample().shape  # (76, 5)
        adj = env.get_adjacency_matrix()

        feature_count = state_shape[1] + 1
        edge_count = state_shape[0]

        self.norm = config['env_config']['norm']
        self.epsilon = config['env_config']['epsilon']

        self.gcn_layers = torch.nn.ModuleList([
            GraphConvolutionLayer(config['model_config']['conv_layers'][i-1]['size'] if i != 0 else feature_count, config['model_config']['conv_layers'][i]['size'], adj)
            for i in range(len(config['model_config']['conv_layers']))
        ])
        self.gcn_layers_activation = [
            getattr(torch.nn.functional, config['model_config']['conv_layers'][i]['activation'])
            for i in range(len(config['model_config']['conv_layers']))
        ]

        self.dense_layers = torch.nn.ModuleList([
            torch.nn.Linear(
                config['model_config']['dense_layers'][i - 1]['size'] if i != 0 else edge_count * config['model_config']['conv_layers'][-1]['size'],
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

        for l in self.gcn_layers:
            torch.nn.init.xavier_uniform_(l.kernel)
            torch.nn.init.zeros_(l.bias)
            torch.nn.init.zeros_(l.beta)

        for l in self.dense_layers:
            torch.nn.init.xavier_uniform_(l.weight)
            torch.nn.init.zeros_(l.bias)

    def forward(self, states, actions):
        actions = self.epsilon * torch.nn.functional.normalize(actions, dim=1, p=self.norm)
        x = torch.cat([states, torch.unsqueeze(actions, 2)], dim=2)
        for a, l in zip(self.gcn_layers_activation, self.gcn_layers):
            x = l(x)
            x = a(x)
        x = x.flatten(start_dim=1)
        for a, l in zip(self.dense_layers_activation, self.dense_layers):
            x = l(x)
            x = a(x)

        x = self.output_layer(x)
        return x


class DoubleDQNModel(torch.nn.Module):
    def __init__(self, device, env, config):
        super(DoubleDQNModel, self).__init__()

        self.q_model = DQNModel(env, config)
        self.target_model = DQNModel(env, config)

        self.norm = config['env_config']['norm']
        self.epsilon = config['env_config']['epsilon']
        self.action_gradient_step_count = config['model_config']['action_gradient_step_count']
        self.action_optimizer = config['model_config']['action_optimizer']

        for target_param, param in zip(self.target_model.parameters(), self.q_model.parameters()):
            target_param.data.copy_(param.data)

        self.tau = config['model_config']['tau']
        self.gamma = config['rl_config']['gamma']
        self.optimizer = getattr(torch.optim, config['model_config']['q_optimizer']['class_name'])(self.q_model.parameters(), **config['model_config']['q_optimizer']['config'])
        self.criterion = getattr(torch.nn, config['model_config']['loss'])()

        self.device = device
        self.to(self.device)

    def forward(self, states, actions):
        return self.target_model.forward(states, actions)

    def max_q(self, states):
        stock_action = torch.distributions.normal.Normal(0, 1).sample(torch.Size((states.shape[0], 76))).to(self.device)
        stock_action.data = self.epsilon * torch.nn.functional.normalize(stock_action, dim=1, p=self.norm)
        initial_q = self.q_model.forward(states, stock_action)
        stock_action.requires_grad = True

        action_optimizer = getattr(torch.optim, self.action_optimizer['class_name'])([stock_action],
                                                                                     **self.action_optimizer['config'])

        for i in range(self.action_gradient_step_count):
            action_optimizer.zero_grad()
            current_q = -self.q_model.forward(states, stock_action)
            current_q.backward()
            action_optimizer.step()
            stock_action.data = self.epsilon * torch.nn.functional.normalize(
                stock_action, dim=1, p=self.norm)

        last_q = self.q_model.forward(states, stock_action)
        return stock_action.cpu().data.numpy(), last_q.mean().cpu().data.numpy(), (last_q - initial_q).mean().cpu().data.numpy()

    def update_target_model(self):
        for target_param, param in zip(self.target_model.parameters(), self.q_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def update(self, states, actions, next_states, rewards, dones, next_actions):

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        rewards = torch.unsqueeze(torch.from_numpy(rewards).float(), dim=1).to(self.device)
        dones = torch.unsqueeze(torch.from_numpy(dones).float(), dim=1).to(self.device)
        next_actions = torch.from_numpy(next_actions).float().to(self.device)

        with torch.no_grad():
            target = rewards + self.gamma * (1 - dones) * self.target_model(next_states, next_actions)

        self.optimizer.zero_grad()
        q_values = self.q_model(states, actions)

        loss = self.criterion(q_values, target)
        loss.backward()
        self.optimizer.step()
        self.update_target_model()

        return dict(
            loss=loss.cpu().data.numpy(),
            r2=r2_score(target, q_values).cpu().data.numpy(),
            max_q=target.max().cpu().data.numpy(),
            min_q=target.min().cpu().data.numpy(),
            mean_q=target.mean().cpu().data.numpy(),
        )


if __name__ == '__main__':

    run_id = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    writer = tb.SummaryWriter(f'logs/{run_id}')
    os.makedirs(f'logs/{run_id}/weights')

    config = dict(
        env_config=dict(
            city='SiouxFalls',
            horizon=50,
            epsilon=30,
            norm=2,
            frac=0.5,
            num_sample=20,
            render_mode=None,
            reward_multiplier=1.0,
            congestion=True,
            trips=dict(
                type='demand_file',
                demand_file='Sirui/traffic_data/sf_demand.txt',
                strategy='random',
                count=10
            ),
            rewarding_rule='vehicle_count',
            observation_type='vector'
        ),
        model_config=dict(
            q_optimizer=dict(
                class_name='Adam',
                config=dict(
                    lr=1e-5
                )
            ),
            loss='MSELoss',
            conv_layers=[
                dict(size=64, activation='relu'),
                dict(size=64, activation='relu'),
                dict(size=64, activation='relu'),
                dict(size=64, activation='relu'),
                dict(size=64, activation='relu'),
                dict(size=16, activation='relu'),
            ],
            dense_layers=[
                dict(size=64, activation='relu'),
                dict(size=64, activation='relu'),
            ],
            tau=0.002,
            action_gradient_step_count=100,
            action_optimizer=dict(
                class_name='Adam',
                config=dict(
                    lr=0.1
                )
            )
        ),
        rl_config=dict(
            noise=dict(
                type='OUActionNoise',
                config=dict(
                    theta=0.15,
                    mean=0,
                    std_deviation=0.2,
                    dt=0.01,
                    target_scale=1.0,
                    anneal=70_000
                )
            ),
            epsilon=dict(
                # type='DecayEpsilon',
                # config=dict(
                #     epsilon_start=1.0,
                #     epsilon_end=0.2,
                #     epsilon_decay=70_000
                # )
                type='ConstantEpsilon',
                config=dict(
                    epsilon=0.95
                )
            ),
            gamma=0.97,
            batch_size=64,
            buffer_size=10_000,
            reward_scale=1.0
        ),
        training_config=dict(
            num_training_per_epoch=256,
            num_episodes=1_000_000,
            # num_training_per_epoch=1,
            # num_episodes=16,
        ),
    )

    with open(f'logs/{run_id}/config.json', 'w') as fd:
        json.dump(config, fd, indent=4)

    env = gym.wrappers.TimeLimit(
        TransportationNetworkEnvironment(config['env_config']),
        max_episode_steps=config['env_config']['horizon']
    )

    device = allocate_best_device()

    model = DoubleDQNModel(device, env, config)

    buffer = ExperienceReplay(config['rl_config']['buffer_size'], config['rl_config']['batch_size'])

    greedy = GreedyRiderVector(config['env_config']['epsilon'], config['env_config']['norm'])

    random = Random(
        env.action_space.shape,
        norm=config['env_config']['norm'],
        epsilon=config['env_config']['epsilon'],
        frac=config['env_config']['frac'],
        selection='discrete'
    )

    noise = getattr(util.rl.exploration, config['rl_config']['noise']['type']) \
            (**config['rl_config']['noise']['config'], shape=env.action_space.sample().shape)

    epsilon = getattr(util.rl.exploration, config['rl_config']['epsilon']['type']) \
            (**config['rl_config']['epsilon']['config'])

    pbar = tqdm(total=config['training_config']['num_episodes'])

    global_step = 0
    total_samples = 0

    for episode in range(config['training_config']['num_episodes']):
        state = env.reset()
        done = False
        truncated = False
        rewards = 0
        step = 0
        discounted_reward = 0
        while not done and not truncated:
            if epsilon():
                action = random.predict(state)
            else:
                action = greedy.predict(state)
                action += noise()

            next_state, reward, done, info = env.step(action)
            truncated = info.get("TimeLimit.truncated", False)
            buffer.add(state, action, reward, next_state, done, next_action=greedy.predict(next_state))
            state = next_state
            rewards += reward
            discounted_reward += reward * (config['rl_config']['gamma'] ** step)
            step += 1
            total_samples += 1

        writer.add_scalar('env/cumulative_reward', rewards, global_step)
        writer.add_scalar('env/discounted_reward', discounted_reward, global_step)
        writer.add_scalar('env/episode_length', step, global_step)

        for _ in range(config['training_config']['num_training_per_epoch']):
            if buffer.size() >= config['rl_config']['batch_size']:
                batch = buffer.sample()
                stats = model.update(*batch)

                writer.add_scalar('model/loss', stats['loss'], global_step)
                writer.add_scalar('model/r2', max(stats['r2'], -1), global_step)

                writer.add_scalar('q/max_q', stats['max_q'], global_step)
                writer.add_scalar('q/mean_q', stats['mean_q'], global_step)
                writer.add_scalar('q/min_q', stats['min_q'], global_step)

                writer.add_scalar('exploration/epsilon', epsilon.get_current_epsilon(), global_step)
                writer.add_scalar('exploration/noise', noise.get_current_noise(), global_step)

                writer.add_scalar('experiences/buffer_size', buffer.size(), global_step)
                writer.add_scalar('experiences/total_samples', total_samples, global_step)

                pbar.set_description(
                    f'Loss {stats["loss"]:.4f} | '
                    f'R2 {stats["r2"]:.6f} | '
                    f'MaxQ {stats["max_q"]:.4f} | '
                    f'MeanQ {stats["mean_q"]:.4f} | '
                    f'MinQ {stats["min_q"]:.4f} | '
                    f'Episode {episode} | '
                    f'Len {step} | '
                    f'CumReward {rewards} | '
                    f'DisReward {discounted_reward:.4f} | '
                    f'Eps {epsilon.get_current_epsilon():.4f} | '
                    f'Noise {noise.get_current_noise():.2f} | '
                    f'ReplayBuffer {buffer.size()}'
                )

            pbar.update(1)
            global_step += 1

            if global_step % 1000 == 0:
                torch.save(model.state_dict(), f'logs/{run_id}/weights/model_{global_step}.pt')

        if episode % 10 == 0:
            truncated = False
            done = False
            step = 0
            rewards = 0
            discounted_reward = 0
            state = env.reset()

            while not done and not truncated:
                action, q_value, improvement = model.max_q(torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0).to(device))
                pbar.set_description(f'Testing {step}, norm: {np.linalg.norm(action[0], ord=config["env_config"]["norm"]):.2f}, improvement: {improvement:.6f}, q_value: {q_value:.6f}')
                next_state, reward, done, info = env.step(action[0])
                truncated = info.get("TimeLimit.truncated", False)
                state = next_state
                rewards += reward
                discounted_reward += reward * (config['rl_config']['gamma'] ** step)
                step += 1

            writer.add_scalar('test/cumulative_reward', rewards, global_step)
            writer.add_scalar('test/discounted_reward', discounted_reward, global_step)
            writer.add_scalar('test/episode_length', step, global_step)

