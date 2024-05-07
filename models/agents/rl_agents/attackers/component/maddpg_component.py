import numpy as np
import torch

from models import CustomModule
from models.agents.rl_agents.attackers.component import ComponentInterface
from util.torch.math import r2_score
from util.torch.misc import hard_sync, soft_sync


class QCritic(CustomModule):

    def __init__(self, name, n_edges, n_features, lr) -> None:
        super().__init__(name)
        self.lr = lr
        self.n_edges = n_edges
        self.n_features = n_features

        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_edges * n_features + n_edges + 1, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, observation, budget, allocation, action):
        return self.model(
            torch.cat(
                (torch.flatten(observation, start_dim=1), budget * allocation, action), dim=1
            )
        )

    def extra_repr(self) -> str:
        return f'n_edges={self.n_edges}, n_features={self.n_features}, lr={self.lr}'


class DeterministicActor(CustomModule):

    def __init__(self, name, n_edges, n_features, lr) -> None:
        super().__init__(name)
        self.lr = lr
        self.n_edges = n_edges
        self.n_features = n_features

        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_edges * n_features + 1, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, n_edges),
            # torch.nn.Softmax(dim=1),
            torch.nn.Softplus()
            # torch.nn.Sigmoid()
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, observation, budget, allocation, deterministic):
        logits = self.model(
            torch.cat(
                (torch.flatten(observation, start_dim=1), allocation * budget), dim=1
            )
        )

        return torch.nn.functional.normalize(logits, dim=1, p=1)

    def extra_repr(self) -> str:
        return f'n_edges={self.n_edges}, n_features={self.n_features}, lr={self.lr}'


class MADDPGComponent(ComponentInterface):
    def __init__(self, edge_component_mapping, n_features, critic_lr, actor_lr, tau, gamma) -> None:
        super().__init__(name='MADDPGComponent')
        self.tau = tau
        self.gamma = gamma
        self.edge_component_mapping = edge_component_mapping
        self.n_components = len(edge_component_mapping)
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr

        self.critics = torch.nn.ModuleList([
            QCritic(f'Critic-{i}', len(edge_component_mapping[i]), n_features, critic_lr) for i
            in
            range(self.n_components)
        ])
        self.actors = torch.nn.ModuleList([
            DeterministicActor(f'Actor-{i}', len(edge_component_mapping[i]), n_features, actor_lr) for i
            in
            range(self.n_components)
        ])

        self.target_critics = torch.nn.ModuleList([
            QCritic(f'Critic-{i}', len(edge_component_mapping[i]), n_features, critic_lr) for i
            in
            range(self.n_components)
        ])
        self.target_actors = torch.nn.ModuleList([
            DeterministicActor(f'Actor-{i}', len(edge_component_mapping[i]), n_features, actor_lr) for i
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
        return self.forward_actor(states, budgets, allocations, deterministic)

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

        return returnable_stats

    def update(self, states, actions, budgets, allocations, next_states, next_budgets, next_allocations, rewards, dones, truncateds):
        return self.__update_multi_agent(
            states, actions, budgets, allocations, next_states, next_budgets, next_allocations, rewards, dones)


class MATD3Component(MADDPGComponent):

    def __init__(self, edge_component_mapping, n_features, critic_lr, actor_lr, tau, gamma, target_action_noise_scale, actor_update_steps) -> None:
        super().__init__(edge_component_mapping, n_features, critic_lr, actor_lr, tau, gamma)

        self.target_action_noise_scale = target_action_noise_scale
        self.actor_update_steps = actor_update_steps

        self.critics_1 = torch.nn.ModuleList([
            QCritic(f'Critic_1-{i}', len(edge_component_mapping[i]), n_features, critic_lr) for i
            in
            range(self.n_components)
        ])

        self.target_critics_1 = torch.nn.ModuleList([
            QCritic(f'Critic_1-{i}', len(edge_component_mapping[i]), n_features, critic_lr) for i
            in
            range(self.n_components)
        ])

        hard_sync(self.target_critics_1, self.critics_1)

        self.training_step = 0

    def forward_critic_1(self, states, budgets, allocations, actions):
        return self.__forward_critic(self.critics_1, states, budgets, allocations, actions)

    def __update_critics(self, index, states, actions, budgets, allocations, next_states, next_budgets, next_allocations, rewards, dones):

        # updating critic

        with torch.no_grad():

            target_action_noise_distribution = torch.distributions.Normal(loc=0.0, scale=self.target_action_noise_scale)

            target_action = self.target_actors[index].forward(
                next_states[:, self.edge_component_mapping[index], :],
                next_budgets,
                next_allocations[:, [index]],
                deterministic=False
            )

            target_action = torch.nn.functional.normalize(
                torch.maximum(target_action + target_action_noise_distribution.sample(target_action.shape), torch.zeros_like(target_action, device=self.device)), p=1, dim=1
            )

            component_target_q_values = self.target_critics[index].forward(
                next_states[:, self.edge_component_mapping[index], :],
                next_budgets,
                next_allocations[:, [index]],
                target_action
            )
            component_target_q_values_1 = self.target_critics_1[index].forward(
                next_states[:, self.edge_component_mapping[index], :],
                next_budgets,
                next_allocations[:, [index]],
                target_action
            )

            component_y = rewards[:, [index]] + self.gamma * torch.minimum(component_target_q_values, component_target_q_values_1) * (1 - dones)

        # Update Critic 0

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

        # Update Critic 1

        component_current_q_1 = self.critics_1[index].forward(
            states[:, self.edge_component_mapping[index], :],
            budgets,
            allocations[:, [index]],
            actions[:, self.edge_component_mapping[index]]
        )

        component_critic_loss_1 = torch.nn.functional.mse_loss(
            component_y,
            component_current_q_1
        )

        self.critics_1[index].optimizer.zero_grad()
        component_critic_loss_1.backward()
        self.critics_1[index].optimizer.step()

        stat = dict(
            q_loss=component_critic_loss.cpu().data.numpy().item(),
            q_r2=max(r2_score(component_y, component_current_q).cpu().data.numpy().item(), -1),
            q_max=component_y.max().cpu().data.numpy().item(),
            q_min=component_y.min().cpu().data.numpy().item(),
            q_mean=component_y.mean().cpu().data.numpy().item(),
        )

        return stat

    def __update_actors(self, index, states, allocations, budgets):
        self.training_step += 1
        if self.training_step % self.actor_update_steps == 0:
            return super().__update_actors(index, states, allocations, budgets)
        return dict()
