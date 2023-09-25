import logging

import torch

from models import BudgetingInterface
from models.dl.noise import OUActionNoise
from util.torch.misc import hard_sync, soft_sync


class BudgetingCritic(torch.nn.Module):
    def __init__(self, n_components, lr):
        super().__init__()
        self.logger = logging.getLogger(name='BudgetingCritic')

        self.state_model = torch.nn.Sequential(
            torch.nn.Linear(5 * n_components, 128),
            torch.nn.ReLU(),
        )

        self.budget_model = torch.nn.Sequential(
            torch.nn.Linear(128 + 1, 1),
            torch.nn.ReLU(),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, aggregated_state, budget):
        return self.budget_model(
            torch.cat((
                self.state_model(torch.flatten(aggregated_state, start_dim=1)),
                budget
            ), dim=1)
        )


class BudgetingActor(torch.nn.Module):
    def __init__(self, n_components, lr):
        super().__init__()
        self.logger = logging.getLogger(name='BudgetingActor')

        self.state_model = torch.nn.Sequential(
            torch.nn.Linear(5 * n_components, 128),
            torch.nn.ReLU(),
        )

        self.allocation_model = torch.nn.Sequential(
            torch.nn.Linear(128 + 1, 2),
            torch.nn.ReLU(),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, aggregated_state, deterministic):
        logits = self.allocation_model(self.state_model(torch.flatten(aggregated_state, start_dim=1)), dim=1)
        locs, scales = torch.split(logits, 1, dim=1)
        dist = torch.distributions.LogNormal(locs, scales)

        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy()


class Budgeting(BudgetingInterface):
    def __init__(self, n_components, actor_lr, critic_lr, tau, gamma, noise):
        super().__init__(name='Budgeting')

        self.n_components = n_components
        self.tau = tau
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.noise = noise

        self.actor = BudgetingActor(n_components, actor_lr)
        self.critic = BudgetingCritic(n_components, critic_lr)
        self.target_actor = BudgetingActor(n_components, actor_lr)
        self.target_critic = BudgetingCritic(n_components, critic_lr)

        hard_sync(self.target_actor, self.actor)
        hard_sync(self.target_critic, self.critic)

    def forward(self, aggregated_state, deterministic):
        action, log_prob, entropy = self.actor.forward(aggregated_state, deterministic=deterministic)
        if not deterministic:
            return torch.maximum(action + self.noise(action.shape), torch.zeros_like(action))
        return action

    def update(self, aggregated_state, budget, reward, next_aggregated_state, done, truncateds):

        # Update critic
        with torch.no_grad():
            next_budget, _, _ = self.target_actor.forward(next_aggregated_state, deterministic=False)
            next_value = self.target_critic.forward(next_aggregated_state, next_budget)
            target = reward + self.gamma * next_value * (1 - done)

        value = self.critic.forward(aggregated_state, budget)
        critic_loss = torch.nn.functional.mse_loss(value, target)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Update actor
        budget, log_prob, entropy = self.actor.forward(aggregated_state, deterministic=False)
        actor_loss = - torch.mean(log_prob * self.critic(aggregated_state, budget))

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Update target networks
        soft_sync(self.target_actor, self.actor, self.tau)
        soft_sync(self.target_critic, self.critic, self.tau)

        return {
            'budgeting/a_val': -actor_loss.item(),
            'budgeting/q_loss': critic_loss.item(),
            'budgeting/a_entropy': entropy.mean().item(),
            'budgeting/noise': self.noise.get_current_noise().detach().cpu().numpy().item(),
        }

    # def get_state_dict(self):
    #     return dict(
    #         n_components=self.n_components,
    #         actor=self.actor.state_dict(),
    #         critic=self.critic.state_dict(),
    #         actor_lr=self.actor_lr,
    #         critic_lr=self.critic_lr,
    #         tau=self.tau,
    #         gamma=self.gamma,
    #         noise=self.noise.get_state_dict(),
    #     )
    #
    # @classmethod
    # def from_state_dict(cls, state_dict):
    #     model = cls(
    #         state_dict['n_components'],
    #         state_dict['actor_lr'],
    #         state_dict['critic_lr'],
    #         state_dict['tau'],
    #         state_dict['gamma'],
    #         OUActionNoise.from_state_dict(state_dict['noise'])
    #     )
    #     model.actor.load_state_dict(state_dict['actor'])
    #     model.critic.load_state_dict(state_dict['critic'])
    #
    #     hard_sync(model.target_actor, model.actor)
    #     hard_sync(model.target_critic, model.critic)
    #     return model