import numpy as np
import torch
import torch.nn

from models import CustomModule, BudgetingInterface
from models.rl_attackers import BaseAttacker
from util.torch.rl import GeneralizedAdvantageEstimation


class StochasticActor(CustomModule):

    def __init__(self, n_features, n_edges, lr):
        super().__init__('Actor')

        self.n_features = n_features
        self.n_edges = n_edges
        self.lr = lr

        self.state_extractor = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(n_features * n_edges, 512),
            torch.nn.ReLU(),
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(512 + 1, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, n_edges),
            torch.nn.Softplus(),
        )

    def extra_repr(self) -> str:
        return f'n_features={self.n_features}, n_edges={self.n_edges}, lr={self.lr}'

    def get_distribution(self, logits):
        return torch.distributions.Dirichlet(concentration=logits + 1)

    def get_logits(self, observation, budget):
        return self.model(
            torch.cat((self.state_extractor(observation), budget), dim=1)
        )

    def forward(self, observation, budget, deterministic):
        logits = self.get_logits(observation, budget)
        distribution = self.get_distribution(logits)
        action = distribution.sample() if not deterministic else distribution.mean
        return action


class VCritic(CustomModule):
    def __init__(self, n_features, n_edges, lr):
        super().__init__('Critic')

        self.n_features = n_features
        self.n_edges = n_edges
        self.lr = lr

        self.state_extractor = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(n_features * n_edges, 512),
            torch.nn.ReLU(),
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(512 + 1, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
        )

    def extra_repr(self) -> str:
        return f'n_features={self.n_features}, n_edges={self.n_edges}, lr={self.lr}'

    def forward(self, observation, budget):
        return self.model(
            torch.cat((self.state_extractor(observation), budget), dim=1)
        )


class FixedBudgetNetworkedWidePPO(BaseAttacker):

    def __init__(self, edge_component_mapping, budgeting: BudgetingInterface, n_features, actor_lr, critic_lr, epsilon, batch_size, gamma, lam, value_coeff, entropy_coeff, n_updates, normalize_advantages) -> None:
        super().__init__('FixedBudgetNetworkedWideDDPG', edge_component_mapping)
        self.budgeting = budgeting

        self.n_features = n_features
        self.n_edges = sum([len(c) for c in edge_component_mapping])
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.lam = lam
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.n_updates = n_updates
        self.batch_size = batch_size
        self.normalize_advantages = normalize_advantages
        self.epsilon = epsilon

        self.actor = StochasticActor(n_features, self.n_edges, actor_lr)
        self.value = VCritic(n_features, self.n_edges, critic_lr)

        self.optimizer = torch.optim.RMSprop([
            {'params': self.actor.parameters(), 'lr': actor_lr},
            {'params': self.value.parameters(), 'lr': critic_lr},
        ])

        self.gae = GeneralizedAdvantageEstimation(gamma=self.gamma, lam=self.lam)

    def extra_repr(self) -> str:
        return f'n_features={self.n_features}, n_edges={self.n_edges}, gamma={self.gamma}, lam={self.lam}, actor_lr={self.actor_lr}, critic_lr={self.critic_lr}, value_coeff={self.value_coeff}, entropy_coeff={self.entropy_coeff}, n_updates={self.n_updates}, normalize_advantages={self.normalize_advantages}'

    def forward(self, observation, deterministic):
        with torch.no_grad():
            aggregated_state = self._aggregate_state(observation)
            budgets = self.budgeting.forward(aggregated_state, deterministic)
            actions = self.actor.forward(observation, budgets, deterministic)

        return actions * budgets, actions, torch.zeros((observation.shape[0], self.n_components)), budgets

    def _update(self, observation, allocations, budgets, action, reward, next_observation, done, truncateds):
        n_samples = observation.shape[0]
        assert n_samples % self.batch_size == 0, 'Batch size must be a multiple of n_samples'

        with torch.no_grad():
            old_values = self.value.forward(observation, budgets)
            advantages = self.gae.forward(old_values, torch.sum(reward, dim=1, keepdim=True), done, truncateds)
            returns = advantages + old_values
            old_logits = self.actor.get_logits(observation, budgets)
            old_distribution = self.actor.get_distribution(old_logits)
            old_log_prob = torch.unsqueeze(old_distribution.log_prob(action), dim=1)

        for epoch in range(self.n_updates):
            permutations = torch.randperm(n_samples)
            for batch in range(n_samples // self.batch_size):
                indices = permutations[batch * self.batch_size: (batch + 1) * self.batch_size]

                values = self.value.forward(observation[indices], budgets[indices])

                value_loss = torch.nn.functional.mse_loss(values, returns[indices])
                if self.normalize_advantages:
                    batch_adv = (advantages[indices] - advantages[indices].mean()) / (advantages[indices].std() + 1e-8)
                else:
                    batch_adv = advantages[indices]
                logits = self.actor.get_logits(observation[indices], budgets[indices])
                distribution = self.actor.get_distribution(logits)
                new_log_prob = torch.unsqueeze(distribution.log_prob(action[indices]), dim=1)
                log_ratio = new_log_prob - old_log_prob[indices]
                surr1 = log_ratio * batch_adv
                surr2 = torch.clamp(log_ratio, np.log(1 - self.epsilon), np.log(1 + self.epsilon)) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                entropy = - distribution.entropy().mean()
                loss = actor_loss + 1.0 * value_loss - self.entropy_coeff * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        with torch.no_grad():
            logits = self.actor.get_logits(observation, budgets)
            distribution = self.actor.get_distribution(logits)
            log_prob = torch.unsqueeze(distribution.log_prob(action), dim=1)
            entropy = distribution.entropy()

        return {
            'attacker/actor_loss': actor_loss.detach().cpu().numpy().item(),
            'attacker/value_loss': value_loss.detach().cpu().numpy().item(),
            'attacker/advantages': advantages.detach().cpu().numpy().mean(),
            'attacker/max_concentration': old_logits.detach().cpu().numpy().max(),
            'attacker/min_concentration': old_logits.detach().cpu().numpy().min(),
            'attacker/max_entropy': entropy.detach().cpu().numpy().max(),
            'attacker/max_log_prob': log_prob.detach().cpu().numpy().max(),
            'attacker/min_log_prob': log_prob.detach().cpu().numpy().min(),
            'attacker/max_log_prob_change': torch.abs(log_prob - old_log_prob).detach().cpu().numpy().max(),
        }

