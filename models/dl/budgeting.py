import logging

import torch

from models import BudgetingInterface, CustomModule
from models.dl.noise import OUActionNoise
from util.torch.math import r2_score
from util.torch.misc import hard_sync, soft_sync


class QCritic(CustomModule):
    def __init__(self, n_components, n_features, lr):
        super().__init__('QCritic')

        self.state_model = torch.nn.Sequential(
            torch.nn.Linear(n_features * n_components, 128),
            torch.nn.ReLU(),
        )

        self.budget_model = torch.nn.Sequential(
            torch.nn.Linear(1, 128),
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(128 + 128, 1),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def extra_repr(self) -> str:
        return f'n_components={self.n_components}, n_features={self.n_features}'

    def forward(self, aggregated_state, budget):
        return self.model(
            torch.cat((
                self.state_model(torch.flatten(aggregated_state, start_dim=1)),
                self.budget_model(budget)
            ), dim=1)
        )


class DeterministicActor(CustomModule):
    def __init__(self, n_components, n_features, lr):
        super().__init__('DeterministicActor')

        self.state_model = torch.nn.Sequential(
            torch.nn.Linear(n_features * n_components, 128),
            torch.nn.ReLU(),
        )

        self.allocation_model = torch.nn.Sequential(
            torch.nn.Linear(128, 1),
            torch.nn.Softplus()
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, aggregated_state, deterministic):
        return self.allocation_model(
            self.state_model(torch.flatten(aggregated_state, start_dim=1))
        )


class DDPGBudgeting(BudgetingInterface):
    def __init__(self, edge_component_mapping, n_features, actor_lr, critic_lr, tau, gamma, noise):
        super().__init__(name='Budgeting')

        self.edge_component_mapping = edge_component_mapping
        self.n_components = len(edge_component_mapping)
        self.n_features = n_features
        self.tau = tau
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.noise = noise

        self.actor = DeterministicActor(self.n_components, n_features, actor_lr)
        self.critic = QCritic(self.n_components, n_features, critic_lr)
        self.target_actor = DeterministicActor(self.n_components, n_features, actor_lr)
        self.target_critic = QCritic(self.n_components, n_features, critic_lr)

        hard_sync(self.target_actor, self.actor)
        hard_sync(self.target_critic, self.critic)

    def forward(self, aggregated_state, deterministic):
        action = self.actor.forward(aggregated_state, deterministic=deterministic)
        if not deterministic:
            return torch.maximum(action + self.noise(action.shape), torch.zeros_like(action))
        return action

    def update(self, aggregated_states, budgets, rewards, next_aggregated_states, dones, truncateds):
        # Update critic
        with torch.no_grad():
            next_budgets = self.target_actor.forward(next_aggregated_states, deterministic=False)
            next_values = self.target_critic.forward(next_aggregated_states, next_budgets)
            target = rewards + self.gamma * next_values * (1 - dones)

        value = self.critic.forward(aggregated_states, budgets)
        critic_loss = torch.nn.functional.mse_loss(value, target)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Update actor
        actions = self.actor.forward(aggregated_states, deterministic=False)
        actor_loss = -self.critic.forward(aggregated_states, actions).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Update target networks
        soft_sync(self.target_actor, self.actor, self.tau)
        soft_sync(self.target_critic, self.critic, self.tau)

        return {
            'budgeting/a_val': -actor_loss.detach().cpu().item(),
            'budgeting/q_loss': critic_loss.detach().cpu().item(),
            'budgeting/max_q': target.max().detach().cpu().item(),
            'budgeting/min_q': target.min().detach().cpu().item(),
            'budgeting/noise': self.noise.get_current_noise().detach().cpu().numpy().item(),
            'budgeting/r2': max(r2_score(target, value).detach().cpu().item(), -1),
        }


class TD3Budgeting(DDPGBudgeting):
    def __init__(self, edge_component_mapping, n_features, actor_lr, critic_lr, tau, gamma, actor_update_steps, target_allocation_noise, noise):
        super().__init__(edge_component_mapping, n_features, actor_lr, critic_lr, tau, gamma, noise)

        self.target_allocation_noise = target_allocation_noise
        self.actor_update_steps = actor_update_steps

        self.critic_1 = QCritic(self.n_components, n_features, critic_lr)
        self.target_critic_1 = QCritic(self.n_components, n_features, critic_lr)

        hard_sync(self.target_critic_1, self.critic_1)

        self.update_iteration = 0

    def update(self, aggregated_states, budgets, rewards, next_aggregated_states, dones, truncateds):
        # Update critic
        with torch.no_grad():
            target_allocation_noise = torch.distributions.Normal(loc=0.0, scale=self.target_allocation_noise)
            next_budgets = self.target_actor.forward(next_aggregated_states, deterministic=False)
            noisy_next_budgets = torch.maximum(next_budgets + target_allocation_noise.sample(next_budgets.shape),
                                               torch.zeros_like(next_budgets, device=self.device))

            next_values = self.target_critic.forward(next_aggregated_states, next_budgets)
            next_values_1 = self.target_critic_1.forward(next_aggregated_states, noisy_next_budgets)
            target = rewards + (1 - dones) * self.gamma * torch.minimum(next_values, next_values_1)

        # Update Critic 0
        value = self.critic.forward(aggregated_states, budgets)
        critic_loss = torch.nn.functional.mse_loss(value, target)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        soft_sync(self.target_critic, self.critic, self.tau)

        # Update Critic 1
        value_1 = self.critic_1.forward(aggregated_states, budgets)
        critic_loss_1 = torch.nn.functional.mse_loss(value_1, target)

        self.critic_1.optimizer.zero_grad()
        critic_loss_1.backward()
        self.critic_1.optimizer.step()
        soft_sync(self.target_critic_1, self.critic_1, self.tau)

        stats = {
            'budgeting/q_loss': critic_loss.detach().cpu().item(),
            'budgeting/max_q': target.max().detach().cpu().item(),
            'budgeting/min_q': target.min().detach().cpu().item(),
            'budgeting/noise': self.noise.get_current_noise().detach().cpu().numpy().item(),
            'budgeting/r2': max(r2_score(target, value).detach().cpu().item(), -1),
        }

        self.update_iteration += 1

        if self.update_iteration % self.actor_update_steps == 0:

            # Update actor
            actions = self.actor.forward(aggregated_states, deterministic=False)
            actor_loss = -self.critic.forward(aggregated_states, actions).mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            soft_sync(self.target_actor, self.actor, self.tau)

            stats['budgeting/a_val'] = -actor_loss.detach().cpu().item()

        return stats
