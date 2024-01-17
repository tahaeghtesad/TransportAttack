import torch

from models import CustomModule
from models.agents.rl_agents.attackers.rl_attackers import BaseAttacker
from util.torch.math import r2_score
from util.torch.misc import hard_sync, soft_sync


class LowLevelQCritic(CustomModule):

    def __init__(self, n_features, n_edges, n_components, lr):
        super().__init__('LowLevelQCritic')
        self.n_features = n_features
        self.n_edges = n_edges
        self.n_components = n_components

        self.state_extractor = torch.nn.Sequential(
            torch.nn.Linear(n_features * n_edges + n_components, 256),
            torch.nn.ReLU(),
        )

        self.action_extractor = torch.nn.Sequential(
            torch.nn.Linear(n_edges, 256)
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(256 + 256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)

    def extra_repr(self) -> str:
        return f'n_features={self.n_features}, n_edges={self.n_edges}, n_components={self.n_components}'

    def forward(self, observation, budgets, actions):
        return self.model(
            torch.cat((
                self.state_extractor(torch.cat((torch.flatten(observation, start_dim=1), budgets), dim=1)),
                self.action_extractor(actions)
            ), dim=1)
        )


class LowLevelDeterministicActor(CustomModule):

    def __init__(self, n_features, n_edges, n_components, lr):
        super().__init__('LowLevelDeterministicActor')
        self.n_features = n_features
        self.n_edges = n_edges
        self.n_components = n_components

        self.state_extractor = torch.nn.Sequential(
            torch.nn.Linear(n_features * n_edges + n_components, 256),
            torch.nn.ReLU(),
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_edges),
            torch.nn.Softplus(),
        )

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)

    def extra_repr(self) -> str:
        return f'n_features={self.n_features}, n_edges={self.n_edges}, n_components={self.n_components}'

    def forward(self, observation, budgets, deterministic=True):
        logits = self.model(
            self.state_extractor(torch.cat((torch.flatten(observation, start_dim=1), budgets), dim=1))
        )

        return torch.nn.functional.normalize(logits, p=1, dim=1)


class HighLevelDeterministicActor(CustomModule):

    def __init__(self, n_features, n_edges, n_components, lr):
        super().__init__('HighLevelDeterministicActor')
        self.n_features = n_features
        self.n_edges = n_edges
        self.n_components = n_components

        self.state_extractor = torch.nn.Sequential(
            torch.nn.Linear(n_features * n_edges, 256),
            torch.nn.ReLU(),
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_components),
            torch.nn.Softplus(),
        )

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)

    def extra_repr(self) -> str:
        return f'n_features={self.n_features}, n_edges={self.n_edges}, n_components={self.n_components}'

    def forward(self, observation, deterministic=True):
        return self.model(
            self.state_extractor(torch.flatten(observation, start_dim=1))
        )


class MixedTD3Attacker(BaseAttacker):

    def __init__(self, edge_component_mapping, n_features, low_level_actor_lr, low_level_critic_lr,
                 high_level_actor_lr, tau, gamma, actor_update_steps, target_noise_scale) -> None:
        super().__init__('MixedTD3Attacker', edge_component_mapping)

        self.models = torch.nn.ModuleDict(
            dict(
                low_level_actors=torch.nn.ModuleList(
                    [LowLevelDeterministicActor(
                        n_features,
                        len(edge_component_mapping[c]),
                        self.n_components,
                        low_level_actor_lr) for c in range(len(edge_component_mapping))]
                ),
                target_low_level_actors=torch.nn.ModuleList(
                    [LowLevelDeterministicActor(
                        n_features,
                        len(edge_component_mapping[c]),
                        self.n_components,
                        low_level_actor_lr) for c in range(len(edge_component_mapping))]
                ),
                low_level_critics_0=torch.nn.ModuleList(
                    [LowLevelQCritic(
                        n_features,
                        len(edge_component_mapping[c]),
                        self.n_components,
                        low_level_critic_lr) for c in range(len(edge_component_mapping))]
                ),
                target_low_level_critics_0=torch.nn.ModuleList(
                    [LowLevelQCritic(
                        n_features,
                        len(edge_component_mapping[c]),
                        self.n_components,
                        low_level_critic_lr) for c in range(len(edge_component_mapping))]
                ),
                low_level_critics_1=torch.nn.ModuleList(
                    [LowLevelQCritic(
                        n_features,
                        len(edge_component_mapping[c]),
                        self.n_components,
                        low_level_critic_lr) for c in range(len(edge_component_mapping))]
                ),
                target_low_level_critics_1=torch.nn.ModuleList(
                    [LowLevelQCritic(
                        n_features,
                        len(edge_component_mapping[c]),
                        self.n_components,
                        low_level_critic_lr) for c in range(len(edge_component_mapping))]
                ),
                high_level_actor=HighLevelDeterministicActor(
                    n_features,
                    self.n_edges,
                    self.n_components,
                    high_level_actor_lr
                ),
                target_high_level_actor=HighLevelDeterministicActor(
                    n_features,
                    self.n_edges,
                    self.n_components,
                    high_level_actor_lr),
            )
        )

        self.target_noise_scale = target_noise_scale
        self.tau = tau
        self.gamma = gamma
        self.actor_update_steps = actor_update_steps

        self.update_step = 0

        hard_sync(self.models['target_low_level_actors'], self.models['low_level_actors'])
        hard_sync(self.models['target_low_level_critics_0'], self.models['low_level_critics_0'])
        hard_sync(self.models['target_low_level_critics_1'], self.models['low_level_critics_1'])
        hard_sync(self.models['target_high_level_actor'], self.models['high_level_actor'])

    def forward(self, observation, deterministic):

        high_level_action = self.models['high_level_actor'](observation, deterministic)

        low_level_action = torch.empty((observation.shape[0], self.n_edges), device=self.device)
        for c in range(self.n_components):
            low_level_action[:, self.edge_component_mapping[c]] = self.models['low_level_actors'][c].forward(
                observation[:, self.edge_component_mapping[c]], high_level_action, deterministic
            )

        budgets = torch.sum(high_level_action, dim=1, keepdim=True)
        allocations = torch.nn.functional.normalize(high_level_action, p=1, dim=1)
        actions = low_level_action
        constructed_actions = self._construct_action(actions, allocations, budgets)

        return constructed_actions, actions, allocations, budgets

    def _update(self, observation, allocations, budgets, action, reward, next_observation, done, truncateds):

        # Updating High Level Critic
        stats = dict()

        with torch.no_grad():
            target_budget_noise_distribution = torch.distributions.Normal(loc=0.0, scale=self.target_noise_scale)
            target_budget = self.models['target_high_level_actor'].forward(next_observation)
            noisy_target_budget = torch.maximum(target_budget + target_budget_noise_distribution.sample(target_budget.shape), torch.zeros_like(target_budget, device=self.device))

        # Updating Low Level Critics

        for c in range(self.n_components):

            with torch.no_grad():
                target_action_noise_distribution = torch.distributions.Normal(loc=0.0, scale=self.target_noise_scale)
                target_action = self.models['target_low_level_actors'][c].forward(next_observation[:, self.edge_component_mapping[c]], noisy_target_budget)
                noisy_target_action = torch.nn.functional.normalize(torch.maximum(target_action + target_action_noise_distribution.sample(target_action.shape), torch.zeros_like(target_action, device=self.device)), dim=1, p=1)

                target_low_level_value_0 = self.models['target_low_level_critics_0'][c].forward(next_observation[:, self.edge_component_mapping[c]], noisy_target_budget, noisy_target_action)
                target_low_level_value_1 = self.models['target_low_level_critics_1'][c].forward(next_observation[:, self.edge_component_mapping[c]], noisy_target_budget, noisy_target_action)

                target_low_level_value = reward[:, [c]] + self.gamma * (1 - done) * torch.minimum(target_low_level_value_0, target_low_level_value_1)

            current_low_level_value_0 = self.models['low_level_critics_0'][c].forward(observation[:, self.edge_component_mapping[c]], allocations * budgets, action[:, self.edge_component_mapping[c]])
            current_low_level_value_1 = self.models['low_level_critics_1'][c].forward(observation[:, self.edge_component_mapping[c]], allocations * budgets, action[:, self.edge_component_mapping[c]])

            high_level_loss_0 = torch.nn.functional.mse_loss(target_low_level_value, current_low_level_value_0)
            high_level_loss_1 = torch.nn.functional.mse_loss(target_low_level_value, current_low_level_value_1)

            self.models['low_level_critics_0'][c].optimizer.zero_grad()
            high_level_loss_0.backward()
            self.models['low_level_critics_0'][c].optimizer.step()

            stats[f'attacker/low_level_{c}/q_loss_0'] = high_level_loss_0.detach().cpu().numpy().item()
            stats[f'attacker/low_level_{c}/q_r2_0'] = max(r2_score(target_low_level_value, current_low_level_value_0).detach().cpu().numpy().item(), -1)
            stats[f'attacker/low_level_{c}/q_max_0'] = target_low_level_value.max().detach().cpu().numpy().item()
            stats[f'attacker/low_level_{c}/q_min_0'] = target_low_level_value.min().detach().cpu().numpy().item()

            self.models['low_level_critics_1'][c].optimizer.zero_grad()
            high_level_loss_1.backward()
            self.models['low_level_critics_1'][c].optimizer.step()

            stats[f'attacker/low_level_{c}/q_loss_1'] = high_level_loss_1.detach().cpu().numpy().item()
            stats[f'attacker/low_level_{c}/q_r2_1'] = max(r2_score(target_low_level_value, current_low_level_value_1).detach().cpu().numpy().item(), -1)
            stats[f'attacker/low_level_{c}/q_max_1'] = target_low_level_value.max().detach().cpu().numpy().item()
            stats[f'attacker/low_level_{c}/q_min_1'] = target_low_level_value.min().detach().cpu().numpy().item()

            soft_sync(self.models['target_low_level_critics_0'][c], self.models['low_level_critics_0'][c], self.tau)
            soft_sync(self.models['target_low_level_critics_1'][c], self.models['low_level_critics_1'][c], self.tau)

        self.update_step += 1

        # Updating Actors
        if self.update_step % self.actor_update_steps == 0:

            # Updating Low Level Actors
            for c in range(self.n_components):
                low_level_action = self.models['low_level_actors'][c].forward(observation[:, self.edge_component_mapping[c]], budgets * allocations)
                low_level_value_0 = self.models['low_level_critics_0'][c].forward(observation[:, self.edge_component_mapping[c]], budgets * allocations, low_level_action)

                low_level_actor_loss = - low_level_value_0.mean()

                self.models['low_level_actors'][c].optimizer.zero_grad()
                low_level_actor_loss.backward()
                self.models['low_level_actors'][c].optimizer.step()

                stats[f'attacker/low_level_{c}/actor_loss'] = low_level_actor_loss.detach().cpu().numpy().item()

                soft_sync(self.models['target_low_level_actors'][c], self.models['low_level_actors'][c], self.tau)

            # Updating High Level Actor
            high_level_action = self.models['high_level_actor'].forward(observation)
            action_values_0 = torch.empty((observation.shape[0], self.n_components), device=self.device)
            for c in range(self.n_components):
                action_values_0[:, [c]] = self.models['low_level_critics_0'][c].forward(observation[:, self.edge_component_mapping[c]], high_level_action, action[:, self.edge_component_mapping[c]])

            high_level_actor_loss = - torch.sum(action_values_0, dim=1, keepdim=True).mean()

            self.models['high_level_actor'].optimizer.zero_grad()
            high_level_actor_loss.backward()
            self.models['high_level_actor'].optimizer.step()

            stats['attacker/high_level_actor_loss'] = high_level_actor_loss.detach().cpu().numpy().item()

            soft_sync(self.models['target_high_level_actor'], self.models['high_level_actor'], self.tau)

        return stats
