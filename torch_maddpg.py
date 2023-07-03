import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils import tensorboard as tb
from tqdm import tqdm

import util.rl.exploration
from attack_heuristics import Random
from transport_env.MultiAgentNetworkEnv import MultiAgentTransportationNetworkEnvironment
from util.rl.experience_replay import ExperienceReplay
from util.torch.math import r2_score


class Critic(torch.nn.Module):

    def __init__(self, config, name, observation_space_shape, action_space_shape) -> None:
        super().__init__()
        self.logger = logging.getLogger(f'{name}')
        self.config = config

        self.observation_in = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(np.prod(observation_space_shape), 64),
            torch.nn.ReLU(),
        )

        self.action_in = torch.nn.Sequential(
            torch.nn.Linear(np.prod(action_space_shape), 64),
            torch.nn.ReLU(),
        )

        self.combined = torch.nn.Sequential(
            torch.nn.Linear(64 + 64 + 1, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            # torch.nn.ReLU(),
        )

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)
        self.gamma = 0.97

    def forward(self, observation, budget, action):
        return self.combined(
            torch.cat(
                (self.observation_in(observation), budget, self.action_in(action)), dim=1
            )
        )


class Actor(torch.nn.Module):
    # input: observation (o)
    # output: A vector of action space shape (a)
    # update rule: should be to maximize the critic value with gradient ascent

    def __init__(self, config, name, observation_space_shape, action_space_shape, budget, norm) -> None:
        super().__init__()
        self.logger = logging.getLogger(f'{name}')
        self.config = config
        self.budget = budget
        self.norm = norm

        self.state_in = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(np.prod(observation_space_shape), 128),
            torch.nn.ReLU(),
        )

        self.output = torch.nn.Sequential(
            torch.nn.Linear(128 + 1, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, np.prod(action_space_shape)),
            torch.nn.Sigmoid()
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, observation, budget):
        logits = self.output(
            torch.cat(
                (self.state_in(observation), budget), dim=1
            )
        )

        logits_norm = torch.linalg.norm(logits, ord=self.norm, dim=1, keepdim=True)
        return self.budget * budget * torch.divide(logits, logits_norm)


class MADDPGModel(torch.nn.Module):
    def __init__(self, env: MultiAgentTransportationNetworkEnvironment, device) -> None:
        super().__init__()
        self.logger = logging.getLogger('MADDPGModel')
        self.n_components = env.n_components
        self.edge_component_mapping = env.edge_component_mapping

        self.critics = torch.nn.ModuleList([
            Critic(config, f'Critic-{i}', env.observation_space[i].shape, env.action_space[i].shape) for i in
            range(self.n_components)
        ])
        self.actors = torch.nn.ModuleList([
            Actor(config, f'Actor-{i}', env.observation_space[i].shape, env.action_space[i].shape, env.budget, env.norm) for i in
            range(self.n_components)
        ])

        self.target_critics = torch.nn.ModuleList([
            Critic(config, f'TargetCritic-{i}', env.observation_space[i].shape, env.action_space[i].shape) for i in
            range(self.n_components)
        ])
        self.target_actors = torch.nn.ModuleList([
            Actor(config, f'TargetActor-{i}', env.observation_space[i].shape, env.action_space[i].shape, env.budget, env.norm) for i in
            range(self.n_components)
        ])

        self.criterion = torch.nn.MSELoss()
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=5e-3)
        self.actor_optimizer = torch.optim.Adam(self.actors.parameters(), lr=1e-3)

        self.logger.info(f'Total parameters: {sum(p.numel() for p in self.parameters())}')

        self.tau = 0.002
        self.gamma = 0.97
        self.device = device
        self.to(self.device)

    def hard_sync(self):
        for critic, target_critic in zip(self.critics, self.target_critics):
            for param, target_param in zip(critic.parameters(), target_critic.parameters()):
                target_param.data.copy_(param.data)

        for actor, target_actor in zip(self.actors, self.target_critics):
            for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                target_param.data.copy_(param.data)

    def soft_sync(self):
        for critic, target_critic in zip(self.critics, self.target_critics):
            for param, target_param in zip(critic.parameters(), target_critic.parameters()):
                target_param.data.copy_(
                    (1 - self.tau) * target_param.data + self.tau * param.data
                )

        for actor, target_actor in zip(self.actors, self.target_actors):
            for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                target_param.data.copy_(
                    (1 - self.tau) * target_param.data + self.tau * param.data
                )

    def update_multi_agent(self, states_dict, actions, next_states_dict, rewards, dones):

        states = torch.from_numpy(np.array([s['feature_vector'] for s in states_dict])).float().to(self.device)
        allocations = torch.from_numpy(np.array([s['allocation'] for s in states_dict])).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).float().to(self.device)

        next_states = torch.from_numpy(np.array([s['feature_vector'] for s in next_states_dict])).float().to(self.device)
        next_allocations = torch.from_numpy(np.array([s['allocation'] for s in next_states_dict])).float().to(self.device)

        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        dones = torch.unsqueeze(torch.from_numpy(np.array(dones)).float(), dim=1).to(self.device)

        q_loss = []
        max_q = []
        min_q = []
        mean_q = []
        r2 = []

        for c in range(self.n_components):
            # Updating component critic
            component_target_action = self.target_actors[c].forward(
                next_states[:, self.edge_component_mapping[c], :], torch.unsqueeze(next_allocations[:, c], dim=1)
            )

            component_target_q_values = self.target_critics[c].forward(
                next_states[:, self.edge_component_mapping[c], :],
                torch.unsqueeze(next_allocations[:, c], dim=1),
                component_target_action
                # next_actions[:, self.edge_component_mapping[c]]
            )

            component_y = torch.unsqueeze(rewards[:, c], dim=1) + self.gamma * component_target_q_values * (1 - dones)
            component_current_q = self.critics[c].forward(
                states[:, self.edge_component_mapping[c], :],
                torch.unsqueeze(allocations[:, c], dim=1),
                actions[:, self.edge_component_mapping[c]]
            )

            component_critic_loss = self.critics[c].criterion(component_y, component_current_q)
            self.critics[c].optimizer.zero_grad()
            component_critic_loss.backward()
            self.critics[c].optimizer.step()

            # updating component actor
            current_component_action = self.actors[c].forward(
                states[:, self.edge_component_mapping[c], :], torch.unsqueeze(allocations[:, c], dim=1)
            )
            component_actor_loss = -torch.mean(
                self.critics[c].forward(
                    states[:, self.edge_component_mapping[c], :], torch.unsqueeze(allocations[:, c], dim=1), current_component_action
                )
            )

            self.actors[c].optimizer.zero_grad()
            component_actor_loss.backward()
            self.actors[c].optimizer.step()

            q_loss.append(component_critic_loss.cpu().data.numpy())
            r2.append(max(r2_score(component_target_q_values, component_current_q).cpu().data.numpy(), -1))
            max_q.append(component_target_q_values.max().cpu().data.numpy())
            min_q.append(component_target_q_values.min().cpu().data.numpy())
            mean_q.append(component_target_q_values.mean().cpu().data.numpy())

        self.soft_sync()

        return dict(
            loss=q_loss,
            min_q=min_q,
            max_q=max_q,
            mean_q=mean_q,
            r2=r2,
        )

    def update_factorized(self, states_dict, actions, next_states_dict, rewards, dones):

        states = torch.from_numpy(np.array([s['feature_vector'] for s in states_dict])).float().to(self.device)
        allocations = torch.from_numpy(np.array([s['allocation'] for s in states_dict])).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).float().to(self.device)

        next_states = torch.from_numpy(np.array([s['feature_vector'] for s in next_states_dict])).float().to(self.device)
        next_allocations = torch.from_numpy(np.array([s['allocation'] for s in next_states_dict])).float().to(self.device)

        rewards = torch.sum(torch.from_numpy(np.array(rewards)).float().to(self.device), dim=1, keepdim=True)
        dones = torch.unsqueeze(torch.from_numpy(np.array(dones)).float(), dim=1).to(self.device)

        # Update critic
        target_actions = self.forward_target_actor(next_states, next_allocations)
        target_q_values = self.forward_target_critic(next_states, next_allocations, target_actions)
        y = rewards + self.gamma * target_q_values * (1 - dones)
        current_q = self.forward_critic(states, allocations, actions)
        critic_loss = self.criterion(y, current_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        current_actions = self.forward_actor(states, allocations)
        actor_loss = -torch.mean(self.forward_critic(states, allocations, current_actions))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        self.soft_sync()

        return dict(
            loss=critic_loss.cpu().data.numpy(),
            actor_loss=actor_loss.cpu().data.numpy(),
            q_values=target_q_values.cpu().data.numpy(),
            r2=r2_score(target_q_values, current_q).cpu().data.numpy(),
            max_q=target_q_values.max().cpu().data.numpy(),
            min_q=target_q_values.min().cpu().data.numpy(),
            mean_q=target_q_values.mean().cpu().data.numpy(),
        )

    def forward_critic(self, states, allocations, actions):
        sum_of_critics = torch.zeros((states.shape[0], 1), device=self.device)
        for c in range(self.n_components):
            sum_of_critics += self.critics[c].forward(
                states[:, self.edge_component_mapping[c], :],
                torch.unsqueeze(allocations[:, c], dim=1),
                actions[:, self.edge_component_mapping[c]]
            )
        return sum_of_critics

    def forward_target_critic(self, states, allocations, actions):
        sum_of_critics = torch.zeros((states.shape[0], 1), device=self.device)
        for c in range(self.n_components):
            sum_of_critics += self.target_critics[c].forward(
                states[:, self.edge_component_mapping[c], :],
                torch.unsqueeze(allocations[:, c], dim=1),
                actions[:, self.edge_component_mapping[c]]
            )
        return sum_of_critics

    def forward_actor(self, states, allocations):
        actions = torch.zeros((states.shape[0], sum([len(c) for c in self.edge_component_mapping])), device=self.device)
        for c in range(self.n_components):
            actions[:, self.edge_component_mapping[c]] = self.actors[c].forward(
                states[:, self.edge_component_mapping[c], :], torch.unsqueeze(allocations[:, c], dim=1))

        return actions

    def forward_target_actor(self, states, allocations):
        actions = torch.zeros((states.shape[0], sum([len(c) for c in self.edge_component_mapping])), device=self.device)
        for c in range(self.n_components):
            actions[:, self.edge_component_mapping[c]] = self.target_actors[c].forward(
                states[:, self.edge_component_mapping[c], :], torch.unsqueeze(allocations[:, c], dim=1))

        return actions


    def forward(self, state):
        states = torch.unsqueeze(torch.from_numpy(state['feature_vector']).float(), dim=0).to(self.device)
        allocations = torch.unsqueeze(torch.from_numpy(state['allocation']).float(), dim=0).to(self.device)

        current_actions = torch.zeros((states.shape[0], sum([len(c) for c in self.edge_component_mapping])),
                                      device=self.device)
        for c in range(self.n_components):
            current_actions[:, self.edge_component_mapping[c]] = self.actors[c].forward(
                states[:, self.edge_component_mapping[c], :], torch.unsqueeze(allocations[:, c], dim=1))
        return current_actions.cpu().data.numpy()[0]


if __name__ == '__main__':
    run_id = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    writer = tb.SummaryWriter(f'logs/{run_id}')
    os.makedirs(f'logs/{run_id}/weights')

    logging.basicConfig(
        stream=sys.stdout,
        format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    logger = logging.getLogger('main')

    config = dict(
        env_config=dict(
            network=dict(
                method='network_file',
                city='SiouxFalls',
                # city='Anaheim',
            ),
            # network=dict(
            #     method='generate',
            #     type='grid',
            #     width=7,
            #     height=7,
            # ),
            # network=dict(
            #     method='generate',
            #     type='line',
            #     num_nodes=10,
            # ),
            # network=dict(
            #     method='generate',
            #     type='cycle',
            #     num_nodes=20,
            # ),
            horizon=50,
            epsilon=30,
            norm=1,
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
            # trips=dict(
            #     type='deterministic',
            #     count=10,
            # ),
            rewarding_rule='travel_time_increased',
            # rewarding_rule='step_count',
            # observation_type='vector',
            norm_penalty_coeff=1.0,
            n_components=4,
            capacity_divider=10000,
        )
    )

    with open(f'logs/{run_id}/config.json', 'w') as fd:
        json.dump(config, fd, indent=4)

    logger.info(config)

    env = MultiAgentTransportationNetworkEnvironment(config['env_config'])
    # env.show_base_graph()

    device = torch.device('cpu')
    logger.info(device)

    model = MADDPGModel(env, device)
    logger.info(model)

    buffer = ExperienceReplay(50_000, 128)

    num_episodes = 100_000

    random = Random(
        action_shape=sum([len(c) for c in env.edge_component_mapping]),
        norm=config['env_config']['norm'],
        epsilon=config['env_config']['epsilon'],
        frac=config['env_config']['frac'],
        selection='discrete'
    )

    epsilon = util.rl.exploration.DecayEpsilon(
        epsilon_start=0.3,
        epsilon_end=0.1,
        epsilon_decay=10_000
    )

    noise = util.rl.exploration.OUActionNoise(
        theta=0.15,
        mean=0.0,
        std_deviation=0.1,
        dt=0.01,
        target_scale=0.005,
        anneal=20_000,
        shape=sum([len(c) for c in env.edge_component_mapping])
    )

    pbar = tqdm(total=num_episodes)
    global_step = 0
    total_samples = 0

    for episode in range(num_episodes):

        should_test = (episode + 1) % 100 == 0

        state = env.reset()
        done = False
        truncated = False
        rewards = 0
        component_rewards = np.zeros(env.n_components)
        step = 0
        discounted_reward = 0
        norm_penalty = 0
        original_reward = 0
        while not done and not truncated:
            if not should_test and epsilon():
                action = random.predict(state)
            else:
                action = model.forward(state)

            if not should_test:
                action += noise()

            next_state, reward, done, info = env.step(action)
            truncated = info.get("TimeLimit.truncated", False)
            buffer.add(state, action, reward, next_state, done)
            norm_penalty += info.get('norm_penalty')
            original_reward += info.get('original_reward')
            state = next_state
            rewards += sum(reward)
            component_rewards += reward
            discounted_reward += sum(reward) * (0.97 ** step)
            step += 1
            total_samples += 1

        target_cat = 'test' if should_test else 'env'
        writer.add_scalar(f'{target_cat}/cumulative_reward', rewards, global_step)
        writer.add_scalar(f'{target_cat}/discounted_reward', discounted_reward, global_step)
        writer.add_scalar(f'{target_cat}/episode_length', step, global_step)
        writer.add_scalar(f'{target_cat}/norm_penalty', norm_penalty / step, global_step)
        writer.add_scalar(f'{target_cat}/original_reward', original_reward, global_step)

        for c in range(env.n_components):
            writer.add_scalar(f'{target_cat}/component_reward/{c}', component_rewards[c], global_step)

        for _ in range(4):

            if buffer.size() >= 128:
                batch = buffer.sample()
                stats = model.update_multi_agent(*batch)

                # writer.add_scalar('model/actor_q', stats['actor_q'], global_step)
                if type(stats['loss']) == list:
                    for c in range(env.n_components):
                        writer.add_scalar(f'model/loss/{c}', stats['loss'][c], global_step)
                        writer.add_scalar(f'model/r2/{c}', max(stats['r2'][c], -1), global_step)

                        writer.add_scalar(f'q/max_q/{c}', stats['max_q'][c], global_step)
                        writer.add_scalar(f'q/mean_q/{c}', stats['mean_q'][c], global_step)
                        writer.add_scalar(f'q/min_q/{c}', stats['min_q'][c], global_step)
                else:
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
                    f'QLoss {["%.2f" % c for c in stats["loss"]]} | ' if type(stats['loss']) == list else f'QLoss {stats["loss"]:.2f} | '
                    f'R2 {["%.2f" % c for c in stats["r2"]]} | ' if type(stats['r2']) == list else f'R2 {stats["r2"]:.2f} | '
                    f'MaxQ {["%.2f" % c for c in stats["max_q"]]} | ' if type(stats['max_q']) == list else f'MaxQ {stats["max_q"]:.2f} | '
                    f'MeanQ {["%.2f" % c for c in stats["mean_q"]]} | ' if type(stats['mean_q']) == list else f'MeanQ {stats["mean_q"]:.2f} | '
                    f'MinQ {["%.2f" % c for c in stats["min_q"]]} | ' if type(stats['min_q']) == list else f'MinQ {stats["min_q"]:.2f} | '
                    f'Episode {episode} | '
                    f'Len {step} | '
                    f'CumReward {rewards:.2f} | '
                    f'DisReward {discounted_reward:.3f} | '
                    f'Eps {epsilon.get_current_epsilon():.3f} | '
                    f'Noise {noise.get_current_noise():.2f} | '
                    f'ReplayBuffer {buffer.size()}'
                )

            pbar.update(1)
            global_step += 1

            if global_step % 1000 == 0:
                torch.save(model.state_dict(), f'logs/{run_id}/weights/model_{global_step}.pt')
