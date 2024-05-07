import torch.nn

import torch

from models import CustomModule
from models.agents import BudgetingInterface
from models.agents.rl_agents.attackers.rl_attackers import BaseAttacker
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
            torch.nn.Linear(n_features * n_edges, 32),
            torch.nn.Tanh(),
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, n_edges),
            torch.nn.Sigmoid()
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def extra_repr(self) -> str:
        return f'n_features={self.n_features}, n_edges={self.n_edges}, lr={self.lr}'

    def forward(self, observation, budget):
        logits = self.model(
            self.state_extractor(observation)
        )
        return torch.nn.functional.normalize(logits, dim=1, p=1)


class QCritic(CustomModule):
    def __init__(self, n_features, n_edges, lr, adj):
        super().__init__('Critic')

        self.n_features = n_features
        self.n_edges = n_edges
        self.lr = lr

        self.state_extractor = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(n_features * n_edges, 32),
            torch.nn.Tanh(),
        )

        self.action_extractor = torch.nn.Sequential(
            torch.nn.Linear(n_edges, 32),
            torch.nn.Tanh(),
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(32 * 2, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.adj = torch.from_numpy(adj).float().to(self.device)

    def extra_repr(self) -> str:
        return f'n_features={self.n_features}, n_edges={self.n_edges}, lr={self.lr}'

    def forward(self, observation, action, budget):
        return self.model(
            torch.cat((self.state_extractor(observation), self.action_extractor(action)), dim=1)
        )# + torch.sum(self.adj * (observation[:, :, [2]] + observation[:, :, [3]]), dim=1)


class FixedBudgetNetworkedWideDDPG(BaseAttacker):

    def __init__(self, adj, edge_component_mapping, budgeting: BudgetingInterface, n_features, actor_lr, critic_lr, gamma,
                 tau) -> None:
        super().__init__('FixedBudgetNetworkedWideDDPG', edge_component_mapping)
        self.budgeting = budgeting

        self.n_features = n_features
        self.n_edges = sum([len(c) for c in edge_component_mapping])
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau

        self.actor = DeterministicActor(n_features, self.n_edges, actor_lr)
        self.target_actor = DeterministicActor(n_features, self.n_edges, actor_lr)

        self.critic = QCritic(n_features, self.n_edges, critic_lr, adj)
        self.target_critic = QCritic(n_features, self.n_edges, critic_lr, adj)

        hard_sync(self.target_critic, self.critic)
        hard_sync(self.target_actor, self.actor)

    def extra_repr(self) -> str:
        return f'n_features={self.n_features}, n_edges={self.n_edges}, gamma={self.gamma}, tau={self.tau}'

    def forward(self, observation, deterministic):
        with torch.no_grad():
            aggregated_state = self._aggregate_state(observation)
            budgets = self.budgeting.forward(aggregated_state, deterministic)
            actions = self.actor.forward(observation, budgets)

        return actions * budgets, actions, torch.ones((observation.shape[0], self.n_components)), budgets

    def _update(self, observation, allocations, budgets, action, reward, next_observation, done, truncateds):
        with torch.no_grad():
            next_aggregated_state = self._aggregate_state(next_observation)
            next_budget = self.budgeting.forward(next_aggregated_state, True)
            next_action = self.target_actor.forward(next_observation, next_budget)

            target_value = torch.sum(reward, dim=1, keepdim=True) + self.gamma * (
                        1 - done) * self.target_critic.forward(next_observation, next_action, next_budget)

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
            'attacker/target_max': target_value.max().cpu().numpy().item(),
            'attacker/target_min': target_value.min().cpu().numpy().item(),
            'attacker/target_mean': target_value.mean().cpu().numpy().item(),
            'attacker/r2': r2_score(target_value, current_value).detach().cpu().numpy().item(),
        }


class FixedBudgetNetworkedWideTD3(FixedBudgetNetworkedWideDDPG):
    def __init__(self, adj, edge_component_mapping, budgeting: BudgetingInterface, n_features, actor_lr, critic_lr, gamma,
                 tau, actor_update_interval, actor_noise) -> None:
        super().__init__(adj, edge_component_mapping, budgeting, n_features, actor_lr, critic_lr, gamma, tau)

        self.actor_update_interval = actor_update_interval
        self.actor_noise = actor_noise

        self.critic_2 = QCritic(n_features, self.n_edges, critic_lr, adj)
        self.target_critic_2 = QCritic(n_features, self.n_edges, critic_lr, adj)

        hard_sync(self.target_critic_2, self.critic_2)

        self.last_actor_update = 0

    def _update(self, observation, allocations, budgets, action, reward, next_observation, done, truncateds):
        with torch.no_grad():
            next_aggregated_state = self._aggregate_state(next_observation)
            next_budget = self.budgeting.forward(next_aggregated_state, True)
            next_action = self.target_actor.forward(next_observation, next_budget)

            noise = torch.normal(0, self.actor_noise, next_action.shape, device=self.device)
            next_noisy_action = torch.nn.functional.normalize(torch.maximum(next_action + noise, torch.zeros_like(next_action, device=self.device)), p=1, dim=1)

            target_value = torch.sum(reward, dim=1, keepdim=True) + self.gamma * (
                    1 - done) * torch.minimum(
                self.target_critic.forward(next_observation, next_noisy_action, next_budget),
                self.target_critic_2.forward(next_observation, next_noisy_action, next_budget)
            )

        current_value_0 = self.critic.forward(observation, action, budgets)
        loss_0 = torch.nn.functional.mse_loss(target_value, current_value_0)

        self.critic.optimizer.zero_grad()
        loss_0.backward()
        self.critic.optimizer.step()

        current_value_1 = self.critic_2.forward(observation, action, budgets)
        loss_1 = torch.nn.functional.mse_loss(target_value, current_value_1)

        self.critic_2.optimizer.zero_grad()
        loss_1.backward()
        self.critic_2.optimizer.step()

        stats = {
            'attacker/q_loss': loss_0.detach().cpu().numpy().item(),
            'attacker/q_loss_1': loss_1.detach().cpu().numpy().item(),
            'attacker/target_max': target_value.max().cpu().numpy().item(),
            'attacker/target_min': target_value.min().cpu().numpy().item(),
            'attacker/target_mean': target_value.mean().cpu().numpy().item(),
            'attacker/r2': r2_score(target_value, current_value_0).detach().cpu().numpy().item(),
        }

        if self.last_actor_update % self.actor_update_interval == 0:

            current_action = self.actor.forward(observation, budgets)
            current_value = self.critic.forward(observation, current_action, budgets)
            actor_loss = - current_value.mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            soft_sync(self.target_actor, self.actor, self.tau)

            stats |= {
                'actor_loss': actor_loss.detach().cpu().numpy().item(),
            }

        self.last_actor_update += 1

        soft_sync(self.target_critic, self.critic, self.tau)
        soft_sync(self.target_critic_2, self.critic_2, self.tau)

        return stats