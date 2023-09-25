import logging

import numpy as np
import torch

from models import ComponentInterface, CustomModule
from util.torch.math import r2_score
from util.torch.misc import hard_sync, soft_sync


class QCritic(CustomModule):

    def __init__(self, name, obs_dim, action_dim, lr) -> None:
        super().__init__(name)
        self.lr = lr

        self.model = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + action_dim + 2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, observation, budget, allocation, action):
        return self.model(
            torch.cat(
                (torch.flatten(observation, start_dim=1), budget, allocation, action), dim=1
            )
        )


class VCritic(torch.nn.Module):
    def __init__(self, name, obs_dim, lr) -> None:
        super().__init__(name)
        self.lr = lr

        self.model = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(obs_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64 + 2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, observation, budget, allocation):
        return self.model(
            torch.cat(
                (observation, budget, allocation), dim=1
            )
        )


class StochasticActor(torch.nn.Module):

    def __init__(self, name, obs_dim, action_dim, lr) -> None:
        super().__init__()
        self.logger = logging.getLogger(f'{name}')

        self.state_in = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(obs_dim, 128),
            torch.nn.ReLU(),
        )

        self.output = torch.nn.Sequential(
            torch.nn.Linear(128 + 2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim),
            torch.nn.ReLU(),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, observation, budget, allocation, deterministic):
        logits = self.output(
            torch.cat(
                (self.state_in(observation), allocation, budget), dim=1
            )
        )
        fixed_logits = logits + 1e-6
        dist = torch.distributions.Dirichlet(concentration=fixed_logits)

        if deterministic:
            sampled_action = dist.mean
        else:
            sampled_action = dist.sample()

        return sampled_action * allocation * budget, dist.log_prob(sampled_action), dist.entropy()


class DeterministicActor(torch.nn.Module):

    def __init__(self, name, obs_dim, action_dim, lr) -> None:
        super().__init__()
        self.logger = logging.getLogger(f'{name}')

        self.model = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + 2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim),
            # torch.nn.Softmax(dim=1),
            # torch.nn.ReLU()
            torch.nn.Sigmoid()
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, observation, budget, allocation, deterministic):
        logits = self.model(
            torch.cat(
                (torch.flatten(observation, start_dim=1), allocation, budget), dim=1
            )
        )

        return torch.nn.functional.normalize(logits, dim=1, p=1) * allocation * budget


class DDPGComponent(ComponentInterface):
    def __init__(self, edge_component_mapping, n_features, critic_lr, actor_lr, tau, gamma, noise) -> None:
        super().__init__(name='DDPGComponent')
        self.tau = tau
        self.gamma = gamma
        self.edge_component_mapping = edge_component_mapping
        self.n_components = len(edge_component_mapping)
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.noise = noise

        self.critics = torch.nn.ModuleList([
            QCritic(f'Critic-{i}', len(edge_component_mapping[i]) * n_features, len(edge_component_mapping[i]), critic_lr) for i
            in
            range(self.n_components)
        ])
        self.actors = torch.nn.ModuleList([
            DeterministicActor(f'Actor-{i}', len(edge_component_mapping[i]) * n_features, len(edge_component_mapping[i]), actor_lr) for i
            in
            range(self.n_components)
        ])

        self.target_critics = torch.nn.ModuleList([
            QCritic(f'Critic-{i}', len(edge_component_mapping[i]) * n_features, len(edge_component_mapping[i]), critic_lr) for i
            in
            range(self.n_components)
        ])
        self.target_actors = torch.nn.ModuleList([
            DeterministicActor(f'Actor-{i}', len(edge_component_mapping[i]) * n_features, len(edge_component_mapping[i]), actor_lr) for i
            in
            range(self.n_components)
        ])

        hard_sync(self.target_critics, self.critics)
        hard_sync(self.target_actors, self.actors)

        self.factorized_critic_optimizer = torch.optim.Adam(self.parameters(), lr=critic_lr)

    def __forward_critic(self, critics, states, budgets, allocations, actions):
        sum_of_critics = torch.empty((states.shape[0], 1), device=self.device)
        for c in range(self.n_components):
            sum_of_critics += critics[c].forward(
                states[:, self.edge_component_mapping[c], :],
                budgets,
                allocations[:, [c]],
                actions[:, self.edge_component_mapping[c]]
            )
        return sum_of_critics

    def forward_critic(self, states, budgets, allocations, actions):
        return self.__forward_critic(self.critics, states, budgets, allocations, actions)

    def forward_target_critic(self, states, budgets, allocations, actions):
        return self.__forward_critic(self.target_critics, states, budgets, allocations, actions)

    def __forward_actor(self, actors, states, budgets, allocations, deterministic):
        actions = torch.empty(
            (states.shape[0], sum([len(c) for c in self.edge_component_mapping])),
            device=self.device)

        for c in range(self.n_components):
            component_action = actors[c].forward(
                states[:, self.edge_component_mapping[c], :],
                budgets,
                allocations[:, [c]],
                deterministic=deterministic
            )
            actions[:, self.edge_component_mapping[c]] = component_action

        return actions

    def forward_actor(self, states, budgets, allocations, deterministic):
        return self.__forward_actor(self.actors, states, budgets, allocations, deterministic)

    def forward_target_actor(self, states, budgets, allocations, deterministic):
        return self.__forward_actor(self.target_actors, states, budgets, allocations, deterministic)

    def forward(self, states, budgets, allocations, deterministic):
        action = self.forward_actor(states, budgets, allocations, deterministic)
        if not deterministic:
            return self.__normalize_action(action + self.noise(action.shape), allocations, budgets)
        else:
            return action

    def __normalize_action(self, action, allocations, budget):
        action = torch.maximum(action, torch.zeros_like(action))
        for c in range(self.n_components):
            action[:, self.edge_component_mapping[c]] = torch.nn.functional.normalize(action[:, self.edge_component_mapping[c]], p=1, dim=1) * allocations[:, [c]] * budget
        return action

    def __update_critics(self, index, states, actions, budgets, allocations, next_states, next_budgets, next_allocations, rewards, dones):

        # updating critic

        with torch.no_grad():
            target_action = self.target_actors[index].forward(
                next_states[:, self.edge_component_mapping[index], :],
                next_budgets,
                next_allocations[:, [index]],
                deterministic=False
            )
            component_target_q_values = self.target_critics[index].forward(
                next_states[:, self.edge_component_mapping[index], :],
                next_budgets,
                next_allocations[:, [index]],
                target_action
            )

            component_y = rewards[:, [index]] + self.gamma * component_target_q_values * (1 - dones)

        component_current_q = self.critics[index].forward(
            states[:, self.edge_component_mapping[index], :],
            budgets,
            allocations[:, [index]],
            actions[:, self.edge_component_mapping[index]]
        )

        component_critic_loss = torch.nn.functional.mse_loss(
            component_y,
            component_current_q
        )
        self.critics[index].optimizer.zero_grad()
        component_critic_loss.backward()
        self.critics[index].optimizer.step()

        stat = dict(
            q_loss=component_critic_loss.cpu().data.numpy().item(),
            q_r2=max(r2_score(component_y, component_current_q).cpu().data.numpy().item(), -1),
            q_max=component_y.max().cpu().data.numpy().item(),
            q_min=component_y.min().cpu().data.numpy().item(),
            q_mean=component_y.mean().cpu().data.numpy().item(),
        )

        return stat

    def __update_actors(self, index, states, allocations, budgets):
        # updating actor

        component_current_action = self.actors[index].forward(
            states[:, self.edge_component_mapping[index], :],
            budgets,
            allocations[:, [index]],
            deterministic=False
        )

        current_value = self.critics[index].forward(
                states[:, self.edge_component_mapping[index], :],
                budgets,
                allocations[:, [index]],
                component_current_action
            )

        component_actor_loss = -torch.mean(
            current_value
        )

        self.actors[index].optimizer.zero_grad()
        component_actor_loss.backward()
        self.actors[index].optimizer.step()

        return dict(
            a_val=-component_actor_loss.cpu().data.numpy().item(),
        )

    def __update_multi_agent(self, states, actions, budgets, allocations, next_states, next_budgets, next_allocations, rewards, dones):
        stats = dict(
            q_loss=np.zeros(self.n_components),
            q_r2=np.zeros(self.n_components),
            q_max=np.zeros(self.n_components),
            q_min=np.zeros(self.n_components),
            q_mean=np.zeros(self.n_components),
            a_val=np.zeros(self.n_components),
        )

        for c in range(self.n_components):
            # Updating component critic

            stat = self.__update_critics(
                c,
                states,
                actions,
                budgets,
                allocations,
                next_states,
                next_budgets,
                next_allocations,
                rewards,
                dones
            )

            for k, v in stat.items():
                stats[k][c] = v

            stat = self.__update_actors(
                c,
                states,
                allocations,
                budgets
            )

            for k, v in stat.items():
                stats[k][c] = v

        soft_sync(self.target_critics, self.critics, self.tau)
        soft_sync(self.target_actors, self.actors, self.tau)

        returnable_stats = dict()
        for k, v in stats.items():
            for c in range(self.n_components):
                returnable_stats[f'component/{k}/{c}'] = v[c]
        return returnable_stats | {
            'component/noise': self.noise.get_current_noise(),
        }

    def update(self, states, actions, budgets, allocations, next_states, next_budgets, next_allocations, rewards, dones):
        return self.__update_multi_agent(
            states, actions, budgets, allocations, next_states, next_budgets, next_allocations, rewards, dones)
