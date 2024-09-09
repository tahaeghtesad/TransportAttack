import torch
import torch
import torch.nn
import torch.nn.utils.prune as prune
from torch import Tensor

from models import CustomModule
from models.agents import BudgetingInterface
from models.agents.rl_agents.attackers.rl_attackers import BaseAttacker
from util.torch.rl import GeneralizedAdvantageEstimation


class MaskedLinear(torch.nn.Linear):

    def __init__(self, mask, in_features: int, out_features: int, skip: bool, **kwargs) -> None:
        super().__init__(in_features, out_features, **kwargs)
        self.mask = mask
        self.skip = skip

    def forward(self, input: Tensor) -> Tensor:
        out = input @ (self.weight * self.mask) + self.bias
        if self.skip:
            return out + input
        return out


class ComponentWiseSparsification(MaskedLinear):

    def __init__(self, edge_component_mapping, n_features, skip=True, **kwargs):
        self.n_edges = sum([len(c) for c in edge_component_mapping])

        self.n_features = n_features
        self.mask = torch.zeros((n_features * self.n_edges, n_features * self.n_edges), dtype=torch.bool)

        for component in edge_component_mapping:
            for i in range(len(component)):
                for j in range(len(component)):
                    self.mask[component[i] * n_features: (component[i] + 1) * n_features, component[j] * n_features: (component[j] + 1) * n_features] = True

        super().__init__(mask=self.mask, in_features=n_features * self.n_edges, out_features=n_features * self.n_edges, skip=skip, **kwargs)

    def extra_repr(self) -> str:
        return f'n_features={self.n_features}, n_edges={self.n_edges}, n_parameters={self.n_features * self.n_edges * self.n_features * self.n_edges}->{self.mask.sum().numpy()}'


class NHopSparsification(MaskedLinear):

    def __init__(self, adj, n_features, hops=1, skip=True, **kwargs):
        self.n_edges = adj.shape[0]
        adj = torch.from_numpy(adj).int()
        assert hops >= 1, f'#hops should be at least 1. {hops}'

        n_hop_adj = torch.zeros((self.n_edges, self.n_edges), dtype=torch.int)

        for _ in range(hops):
            n_hop_adj = torch.logical_or((adj @ n_hop_adj) > 0, adj)

        self.n_features = n_features
        self.mask = torch.zeros((n_features * self.n_edges, n_features * self.n_edges), dtype=torch.bool)

        for i in range(self.n_edges):
            for j in range(self.n_edges):
                if n_hop_adj[i, j] == 1:
                    self.mask[i * n_features: (i + 1) * n_features, j * n_features: (j + 1) * n_features] = True

        super().__init__(mask=self.mask, in_features=n_features * self.n_edges, out_features=n_features * self.n_edges, skip=skip, **kwargs)

    def extra_repr(self) -> str:
        return f'n_features={self.n_features}, n_edges={self.n_edges}, n_parameters={self.n_features * self.n_edges * self.n_features * self.n_edges}->{self.mask.sum().numpy()}'


class StochasticActor(CustomModule):

    def __init__(self, adj, edge_component_mapping, n_features, state_extractor=None):
        super().__init__('Actor')

        self.n_features = n_features
        self.n_edges = sum([len(c) for c in edge_component_mapping])
        self.edge_component_mapping = edge_component_mapping

        self.state_extractor = torch.nn.Sequential(
            torch.nn.Flatten(),
            NHopSparsification(adj, n_features, 1, skip=True),
            NHopSparsification(adj, n_features, 1, skip=True),
            NHopSparsification(adj, n_features, 1, skip=True),
            # ComponentWiseSparsification(edge_component_mapping=edge_component_mapping, n_features=n_features),
            # torch.nn.Linear(self.n_features * self.n_edges, self.n_features * self.n_edges),
            torch.nn.ReLU(),
        ) if state_extractor is None else state_extractor

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.n_features * self.n_edges, self.n_edges),
        )

    def extra_repr(self) -> str:
        return f'n_features={self.n_features}, n_edges={self.n_edges}'

    def get_distribution(self, logits):
        return torch.distributions.Dirichlet(concentration=torch.exp(logits) + 1.0)

    def get_logits(self, observation, budget):
        return self.model(
            self.state_extractor(observation)
        )

    def get_log_prob(self, logits, action):
        distribution = self.get_distribution(logits)
        return distribution.log_prob(action).unsqueeze(1)

    def get_entropy(self, logits):
        distribution = self.get_distribution(logits)
        return distribution.entropy().unsqueeze(1)

    def forward(self, observation, budget, deterministic):
        logits = self.get_logits(observation, budget)
        distribution = self.get_distribution(logits)
        action = distribution.sample() if not deterministic else distribution.mean
        return action


class VCritic(CustomModule):
    def __init__(self, adj, edge_component_mapping, n_features, state_extractor=None):
        super().__init__('Critic')

        self.n_features = n_features
        self.n_edges = sum([len(c) for c in edge_component_mapping])
        self.edge_component_mapping = edge_component_mapping

        self.state_extractor = torch.nn.Sequential(
            torch.nn.Flatten(),
            # ComponentWiseSparsification(edge_component_mapping=edge_component_mapping, n_features=n_features),
            # torch.nn.Linear(self.n_features * self.n_edges, self.n_features * self.n_edges),
            NHopSparsification(adj, n_features, 1, skip=True),
            NHopSparsification(adj, n_features, 1, skip=True),
            NHopSparsification(adj, n_features, 1, skip=True),
            torch.nn.ReLU(),
        ) if state_extractor is None else state_extractor

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.n_edges * self.n_features, 1),
        )

    def extra_repr(self) -> str:
        return f'n_features={self.n_features}, n_edges={self.n_edges}'

    def forward(self, observation, budget):
        return self.model(
            self.state_extractor(observation)
        )


class FixedBudgetNetworkedWidePPO(BaseAttacker):

    def __init__(self, adj, edge_component_mapping, budgeting: BudgetingInterface, n_features, actor_lr, critic_lr,
                 log_epsilon, batch_size, gamma, lam, value_coeff, entropy_coeff, n_updates, warm_up, normalize_advantages, share_parameters=False) -> None:
        super().__init__('FixedBudgetNetworkedWideDDPG', edge_component_mapping)
        self.budgeting = budgeting

        self.n_features = n_features
        self.adj = adj
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
        self.log_epsilon = log_epsilon
        self.warm_up = warm_up
        self.update_step = 0

        state_extractor = torch.nn.Sequential(
            torch.nn.Flatten(),
            ComponentWiseSparsification(edge_component_mapping=edge_component_mapping, n_features=n_features),
            torch.nn.ReLU(),
        ) if share_parameters is True else None

        self.actor = StochasticActor(adj=adj, edge_component_mapping=edge_component_mapping, n_features=n_features, state_extractor=state_extractor)
        self.value = VCritic(adj=adj, edge_component_mapping=edge_component_mapping, n_features=n_features, state_extractor=state_extractor)

        optimizer_params = [
            {'params': self.actor.model.parameters(), 'lr': actor_lr},
            {'params': self.value.model.parameters(), 'lr': critic_lr},
            {'params': state_extractor.parameters(), 'lr': actor_lr},
        ] if share_parameters else [
            {'params': self.actor.parameters(), 'lr': actor_lr},
            {'params': self.value.parameters(), 'lr': critic_lr},
        ]

        self.optimizer = torch.optim.AdamW(
            optimizer_params
        )

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
        self.update_step += 1
        n_samples = observation.shape[0]
        assert n_samples % self.batch_size == 0, 'Batch size must be a multiple of n_samples'

        with torch.no_grad():
            old_values = self.value.forward(observation, budgets)
            old_next_values = self.value.forward(next_observation, budgets)
            advantages = self.gae.forward(old_values, torch.sum(reward, dim=1, keepdim=True), done, truncateds, old_next_values)
            returns = advantages + old_values
            old_logits = self.actor.get_logits(observation, budgets)
            old_log_prob = self.actor.get_log_prob(old_logits, action)
            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std())

        for epoch in range(self.n_updates):
            permutations = torch.randperm(n_samples)
            for batch in range(n_samples // self.batch_size):
                indices = permutations[batch * self.batch_size: (batch + 1) * self.batch_size]

                values = self.value.forward(observation[indices], budgets[indices])

                value_loss = torch.nn.functional.mse_loss(values, returns[indices])
                logits = self.actor.get_logits(observation[indices], budgets[indices])
                new_entropy = - self.actor.get_entropy(logits).mean()
                new_log_prob = self.actor.get_log_prob(logits, action[indices])
                log_ratio = new_log_prob - old_log_prob[indices]
                surr1 = log_ratio * advantages[indices]
                surr2 = torch.clamp(log_ratio, -self.log_epsilon, self.log_epsilon) * advantages[indices]
                actor_loss = -torch.min(surr1, surr2).mean()

                if self.update_step < self.warm_up:
                    loss = value_loss
                else:
                    loss = actor_loss + 1.0 * value_loss + self.entropy_coeff * torch.exp(logits).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        with torch.no_grad():
            logits = self.actor.get_logits(observation, budgets)
            log_prob = self.actor.get_log_prob(logits, action)
            entropy = self.actor.get_entropy(logits)

        return {
            'attacker/actor_loss': actor_loss.detach().cpu().numpy().item(),
            'attacker/value_loss': value_loss.detach().cpu().numpy().item(),
            'attacker/advantages': advantages.detach().cpu().numpy().mean(),
            'attacker/max_concentration': logits.detach().cpu().numpy().max(),
            'attacker/mean_concentration': logits.detach().cpu().numpy().mean(),
            'attacker/max_log_prob': log_prob.detach().cpu().numpy().max(),
            'attacker/min_log_prob': log_prob.detach().cpu().numpy().min(),
            'attacker/max_log_prob_change': (log_prob - old_log_prob).detach().cpu().numpy().max(),
            'attacker/min_log_prob_change': (log_prob - old_log_prob).detach().cpu().numpy().min(),
            'attacker/mean_entropy': entropy.detach().cpu().numpy().mean(),
        }
