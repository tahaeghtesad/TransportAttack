import logging

import numpy as np
import torch

from transport_env.MultiAgentNetworkEnv import MultiAgentTransportationNetworkEnvironment
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
    def __init__(self, env: MultiAgentTransportationNetworkEnvironment, config, device) -> None:
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