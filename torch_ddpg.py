import json
import os
from datetime import datetime

import torch
from torch.utils import tensorboard as tb
from tqdm import tqdm

import util.rl.exploration
from attack_heuristics import Random
from transport_env.NetworkEnv import TransportationNetworkEnvironment
from util.rl.experience_replay import ExperienceReplay
from util.torch.gcn import GraphConvolutionResidualBlock
from util.torch.math import r2_score
from util.torch.misc import allocate_best_device


class DDPGCritic(torch.nn.Module):
    def __init__(self, env, config):
        super(DDPGCritic, self).__init__()
        action_shape = env.action_space.sample().shape  # (76, )
        state_shape = env.observation_space.sample().shape  # (76, 5)
        adj = env.get_adjacency_matrix()

        edge_count = state_shape[0]

        if len(config['model_config']['conv_layers']) > 0:
            self.gcn_layers = torch.nn.ModuleList([
                GraphConvolutionResidualBlock(
                    state_shape[1] + 1,
                    adj,
                    config['model_config']['conv_layers'][i]['activation'],
                    config['model_config']['conv_layers'][i]['depth']
                )
                for i in range(len(config['model_config']['conv_layers']))
            ])

        self.dense_layers = torch.nn.ModuleList([
            torch.nn.Linear(
                config['model_config']['dense_layers'][i - 1]['size'] if i != 0 else state_shape[0] * (state_shape[1] + 1),
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

    def forward(self, states, actions):
        x = torch.cat([states, torch.unsqueeze(actions, 2)], dim=2)
        if hasattr(self, 'gcn_layers'):
            for l in self.gcn_layers:
                x = l(x)

        x = x.flatten(start_dim=1)
        for a, l in zip(self.dense_layers_activation, self.dense_layers):
            x = l(x)
            x = a(x)

        x = torch.nn.functional.relu(self.output_layer(x))
        return x


class DDPGActor(torch.nn.Module):
    def __init__(self, env, config):
        super(DDPGActor, self).__init__()

        action_shape = env.action_space.sample().shape  # (76, )
        state_shape = env.observation_space.sample().shape  # (76, 5)
        adj = env.get_adjacency_matrix()

        edge_count = state_shape[0]

        self.norm = config['env_config']['norm']
        self.epsilon = config['env_config']['epsilon']

        if len(config['model_config']['conv_layers']) > 0:
            self.gcn_layers = torch.nn.ModuleList([
                GraphConvolutionResidualBlock(
                    state_shape[1],
                    adj,
                    config['model_config']['conv_layers'][i]['activation'],
                    config['model_config']['conv_layers'][i]['depth']
                )
                for i in range(len(config['model_config']['conv_layers']))
            ])

        self.dense_layers = torch.nn.ModuleList([
            torch.nn.Linear(
                config['model_config']['dense_layers'][i - 1]['size'] if i != 0 else state_shape[0] * state_shape[1],
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
            action_shape[0]
        )

        for l in self.dense_layers:
            torch.nn.init.xavier_uniform_(l.weight)
            torch.nn.init.zeros_(l.bias)

    def forward(self, states):
        x = states
        if hasattr(self, 'gcn_layers'):
            for l in self.gcn_layers:
                x = l(x)

        x = x.flatten(start_dim=1)
        for a, l in zip(self.dense_layers_activation, self.dense_layers):
            x = a(l(x))

        x = torch.nn.functional.relu(self.output_layer(x))
        x = torch.nn.functional.normalize(x, dim=1, p=self.norm) * self.epsilon
        return x

class DDPGModel(torch.nn.Module):
    def __init__(self, device, env, config):
        super(DDPGModel, self).__init__()

        self.critic = DDPGCritic(env, config)
        self.target_critic = DDPGCritic(env, config)
        self.actor = DDPGActor(env, config)
        self.target_actor = DDPGActor(env, config)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.tau = config['model_config']['tau']
        self.gamma = config['rl_config']['gamma']
        self.critic_optimizer = getattr(torch.optim, config['model_config']['critic_optimizer']['class_name'])(self.critic.parameters(), **config['model_config']['critic_optimizer']['config'])
        self.actor_optimizer = getattr(torch.optim, config['model_config']['actor_optimizer']['class_name'])(self.actor.parameters(), **config['model_config']['actor_optimizer']['config'])
        self.criterion = getattr(torch.nn, config['model_config']['loss'])()

        self.device = device
        self.to(self.device)

    def forward(self, states):
        return self.actor(states)

    def update_target_model(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def update(self, states, actions, next_states, rewards, dones):

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        rewards = torch.unsqueeze(torch.from_numpy(rewards).float(), dim=1).to(self.device)
        dones = torch.unsqueeze(torch.from_numpy(dones).float(), dim=1).to(self.device)

        with torch.no_grad():
            target = rewards + self.gamma * (1 - dones) * self.target_critic(next_states, self.target_actor(next_states))
        q_values = self.critic(states, actions)
        loss = self.criterion(q_values, target)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_target_model()

        return dict(
            loss=loss.cpu().data.numpy(),
            r2=r2_score(target, q_values).cpu().data.numpy(),
            max_q=target.max().cpu().data.numpy(),
            min_q=target.min().cpu().data.numpy(),
            mean_q=target.mean().cpu().data.numpy(),
            actor_q=-actor_loss.cpu().data.numpy()
        )


if __name__ == '__main__':

    run_id = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    writer = tb.SummaryWriter(f'logs/{run_id}')
    os.makedirs(f'logs/{run_id}/weights')


    config = dict(
        env_config=dict(
            # network=dict(
            #     method='network_file',
            #     city='SiouxFalls',
            # ),
            # network=dict(
            #     method='generate',
            #     type='grid',
            #     width=5,
            #     height=3,
            # ),
            # network=dict(
            #     method='generate',
            #     type='line',
            #     num_nodes=10,
            # ),
            network=dict(
                method='generate',
                type='cycle',
                num_nodes=20,
            ),
            # city='SiouxFalls',
            horizon=50,
            epsilon=30,
            norm=2,
            frac=0.5,
            num_sample=20,
            render_mode=None,
            reward_multiplier=1.0,
            congestion=True,
            # trips=dict(
            #     type='demand_file',
            #     demand_file='Sirui/traffic_data/sf_demand.txt',
            #     strategy='top',
            #     count=10
            # ),
            trips=dict(
                type='deterministic',
                count=10,
            ),
            rewarding_rule='travel_time_increased',
            # rewarding_rule='step_count',
            observation_type='vector',
            norm_penalty_coeff=0.01,
        ),
        model_config=dict(
            critic_optimizer=dict(
                class_name='Adam',
                config=dict(
                    lr=2.5e-4
                )
            ),
            actor_optimizer=dict(
                class_name='Adam',
                config=dict(
                    lr=2.5e-4
                )
            ),
            loss='MSELoss',
            conv_layers=[
                # dict(depth=3, activation='relu'),
                # dict(depth=3, activation='relu'),
            ],
            dense_layers=[
                dict(size=512, activation='relu'),
                dict(size=512, activation='relu'),
            ],
            tau=0.002,
        ),
        rl_config=dict(
            noise=dict(
                type='OUActionNoise',
                config=dict(
                    theta=0.15,
                    mean=0,
                    std_deviation=0.1,
                    dt=0.01,
                    target_scale=0.005,
                    anneal=20_000
                )
            ),
            epsilon=dict(
                type='DecayEpsilon',
                config=dict(
                    epsilon_start=0.3,
                    epsilon_end=0.01,
                    epsilon_decay=10_000,
                )
            ),
            gamma=0.97,
            batch_size=64,
            buffer_size=50_000,
            reward_scale=1.0
        ),
        training_config=dict(
            num_training_per_epoch=64,
            num_episodes=1_000_000,
        ),
    )
    with open(f'logs/{run_id}/config.json', 'w') as fd:
        json.dump(config, fd, indent=4)

    env = TransportationNetworkEnvironment(config['env_config'])

    device = allocate_best_device('cpu')

    model = DDPGModel(device, env, config)
    print(config)
    print(model)
    print(f'Total number of edges: {env.base.number_of_edges()}')

    buffer = ExperienceReplay(config['rl_config']['buffer_size'], config['rl_config']['batch_size'])

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

        should_test = (episode + 1) % 100 == 0

        state = env.reset()
        done = False
        truncated = False
        rewards = 0
        step = 0
        discounted_reward = 0
        norm_penalty = 0
        original_reward = 0
        while not done and not truncated:
            if not should_test and epsilon():
                action = random.predict(state)
            else:
                tensored_state = torch.unsqueeze(torch.from_numpy(state), 0).float().to(device)
                action = model.forward(tensored_state)
                action = action.cpu().data.numpy()[0]

            if not should_test:
                action += noise()

            next_state, reward, done, info = env.step(action)
            truncated = info.get("TimeLimit.truncated", False)
            buffer.add(state, action, reward, next_state, done)
            norm_penalty += info.get('norm_penalty')
            original_reward += info.get('original_reward')
            state = next_state
            rewards += reward
            discounted_reward += reward * (config['rl_config']['gamma'] ** step)
            step += 1
            total_samples += 1

        target_cat = 'test' if should_test else 'env'
        writer.add_scalar(f'{target_cat}/cumulative_reward', rewards, global_step)
        writer.add_scalar(f'{target_cat}/discounted_reward', discounted_reward, global_step)
        writer.add_scalar(f'{target_cat}/episode_length', step, global_step)
        writer.add_scalar(f'{target_cat}/norm_penalty', norm_penalty / step, global_step)
        writer.add_scalar(f'{target_cat}/original_reward', original_reward, global_step)

        for _ in range(config['training_config']['num_training_per_epoch']):

            writer.add_scalar('constants/batch_size', config['rl_config']['batch_size'], global_step)
            writer.add_scalar('constants/gamma', config['rl_config']['gamma'], global_step)
            writer.add_scalar('constants/actor_lr', config['model_config']['actor_optimizer']['config']['lr'], global_step)
            writer.add_scalar('constants/critic_lr', config['model_config']['critic_optimizer']['config']['lr'], global_step)
            writer.add_scalar('constants/tau', config['model_config']['tau'], global_step)
            writer.add_scalar('constants/num_training_per_epoch', config['training_config']['num_training_per_epoch'], global_step)
            writer.add_scalar('constants/env/trip_count', config['env_config']['trips']['count'], global_step)

            if buffer.size() >= config['rl_config']['batch_size']:
                batch = buffer.sample()
                stats = model.update(*batch)

                writer.add_scalar('model/actor_q', stats['actor_q'], global_step)
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
                    f'ActorQ {stats["actor_q"]:.4f} | '
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
