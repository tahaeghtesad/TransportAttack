import torch.nn

from models import CustomModule, BudgetingInterface, NoiseInterface
from models.rl_attackers import BaseAttacker
import torch

from util.torch.math import r2_score
from util.torch.misc import hard_sync, soft_sync


class DeterministicActor(CustomModule):

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
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def extra_repr(self) -> str:
        return f'n_features={self.n_features}, n_edges={self.n_edges}, lr={self.lr}'

    def forward(self, observation, budget):
        logits = self.model(
            torch.cat((self.state_extractor(observation), budget), dim=1)
        )
        return torch.nn.functional.normalize(logits)


class QCritic(CustomModule):
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

        self.action_extractor = torch.nn.Sequential(
            torch.nn.Linear(n_edges, 512),
            torch.nn.ReLU(),
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(512 * 2 + 1, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def extra_repr(self) -> str:
        return f'n_features={self.n_features}, n_edges={self.n_edges}, lr={self.lr}'

    def forward(self, observation, action, budget):
        return self.model(
            torch.cat((self.state_extractor(observation), self.action_extractor(action), budget), dim=1)
        )


class FixedBudgetNetworkedWideDDPG(BaseAttacker):

    def __init__(self, edge_component_mapping, budgeting: BudgetingInterface, n_features, actor_lr, critic_lr, gamma, tau, noise: NoiseInterface) -> None:
        super().__init__('FixedBudgetNetworkedWideDDPG', edge_component_mapping)
        self.budgeting = budgeting

        self.n_features = n_features
        self.n_edges = sum([len(c) for c in edge_component_mapping])
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.noise = noise

        self.actor = DeterministicActor(n_features, self.n_edges, actor_lr)
        self.target_actor = DeterministicActor(n_features, self.n_edges, actor_lr)

        self.critic = QCritic(n_features, self.n_edges, critic_lr)
        self.target_critic = QCritic(n_features, self.n_edges, critic_lr)

        hard_sync(self.target_critic, self.critic)
        hard_sync(self.target_actor, self.actor)

    def extra_repr(self) -> str:
        return f'n_features={self.n_features}, n_edges={self.n_edges}, gamma={self.gamma}, tau={self.tau}'

    def forward(self, observation, deterministic):
        with torch.no_grad():
            aggregated_state = self._aggregate_state(observation)
            budgets = self.budgeting.forward(aggregated_state, deterministic)
            actions = self.actor.forward(observation, budgets)

        if not deterministic:
            actions = torch.nn.functional.normalize(torch.maximum(actions + self.noise(actions.shape), torch.zeros_like(actions, device=self.device)))

        return actions * budgets, actions, torch.zeros((observation.shape[0], self.n_components)), budgets

    def _update(self, observation, allocations, budgets, action, reward, next_observation, done, truncateds):

        with torch.no_grad():
            next_aggregated_state = self._aggregate_state(next_observation)
            next_budget = self.budgeting.forward(next_aggregated_state, True)
            next_action = self.target_actor.forward(next_observation, next_budget)

            target_value = torch.sum(reward, dim=1, keepdim=True) + self.gamma * (1 - done) * self.target_critic.forward(next_observation, next_action, next_budget)

        current_value = self.critic.forward(observation, action, budgets)
        loss = torch.nn.functional.mse_loss(target_value, current_value)

        self.critic.optimizer.zero_grad()
        loss.backward()
        self.critic.optimizer.step()

        current_action = self.actor.forward(observation, budgets)
        current_value = self.critic.forward(observation, current_action, budgets)
        actor_loss = - current_value.mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        soft_sync(self.target_critic, self.critic, self.tau)
        soft_sync(self.target_actor, self.actor, self.tau)

        return {
            'attacker/q_loss': loss.detach().cpu().numpy().item(),
            'attacker/q_max': target_value.max().cpu().numpy().item(),
            'attacker/q_min': target_value.min().cpu().numpy().item(),
            'attacker/q_mean': target_value.mean().cpu().numpy().item(),
            'attacker/r2': r2_score(target_value, current_value).detach().cpu().numpy().item(),
            'attacker/noise': self.noise.get_current_noise().detach().cpu().numpy().item()
        }
