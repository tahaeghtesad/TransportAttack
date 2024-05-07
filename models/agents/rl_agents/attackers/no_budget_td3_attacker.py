import torch

from models import CustomModule
from models.agents.rl_agents.attackers.rl_attackers import BaseAttacker
from util.torch.misc import hard_sync, soft_sync


class NoBudgetQCriticWithBiasEstimator(CustomModule):

    def __init__(self, n_features, n_edges, lr, adj):
        super().__init__('QCriticWithBiasEstimator')
        self.n_features = n_features
        self.n_edges = n_edges
        self.lr = lr
        self.adj = torch.from_numpy(adj).float().to(self.device)
        assert adj.shape == (n_edges, 1), f'adj shape {adj.shape} does not match n_edges ({n_edges}, 1)'

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
            torch.nn.Linear(512 * 2, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
        )

        self.bias_estimator = torch.nn.Sequential(
            torch.nn.Linear(n_edges * 2, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
        )

        self.adj_estimator = torch.nn.Parameter(self.adj, requires_grad=True)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def extra_repr(self) -> str:
        return f'n_features={self.n_features}, n_edges={self.n_edges}, lr={self.lr}, adj={self.adj}'

    def forward(self, observation, action):
        return self.model(
            torch.cat((self.state_extractor(observation), self.action_extractor(action)), dim=1)
        ) + torch.sum(self.adj * (observation[:, :, [2]] + observation[:, :, [3]]), dim=1)
        # + torch.sum(self.adj_estimator * (observation[:, :, [2]] + observation[:, :, [3]]), dim=1))
        # + self.bias_estimator(torch.flatten(observation[:, :, [2, 3]], start_dim=1))


class DeterministicNoBudgetActor(CustomModule):
    def __init__(self, n_features, n_edges, lr):
        super().__init__('DeterministicNoBudgetActor')

        self.n_features = n_features
        self.n_edges = n_edges
        self.lr = lr

        self.state_extractor = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(n_features * n_edges, 512),
            torch.nn.ReLU(),
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, n_edges),
            torch.nn.ReLU(),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def extra_repr(self) -> str:
        return f'n_features={self.n_features}, n_edges={self.n_edges}, lr={self.lr}'

    def forward(self, observation):
        return self.model(self.state_extractor(observation))


class NoBudgetTD3Attacker(BaseAttacker):
    def __init__(self, edge_component_mapping, n_features, adj, actor_lr, critic_lr, gamma, tau, target_noise_scale, actor_update_steps):
        super().__init__('NoBudgetTD3Attacker', edge_component_mapping=edge_component_mapping)

        self.actor = DeterministicNoBudgetActor(n_features, sum([len(v) for v in edge_component_mapping]), actor_lr)
        self.target_actor = DeterministicNoBudgetActor(n_features, sum([len(v) for v in edge_component_mapping]), actor_lr)

        self.critic = NoBudgetQCriticWithBiasEstimator(n_features, sum([len(v) for v in edge_component_mapping]), critic_lr, adj)
        self.target_critic = NoBudgetQCriticWithBiasEstimator(n_features, sum([len(v) for v in edge_component_mapping]), critic_lr, adj)

        self.critic_1 = NoBudgetQCriticWithBiasEstimator(n_features, sum([len(v) for v in edge_component_mapping]), critic_lr, adj)
        self.target_critic_1 = NoBudgetQCriticWithBiasEstimator(n_features, sum([len(v) for v in edge_component_mapping]), critic_lr, adj)

        self.n_components = len(edge_component_mapping)
        self.gamma = gamma
        self.tau = tau
        self.target_noise_scale = target_noise_scale
        self.actor_update_steps = actor_update_steps

        hard_sync(self.target_actor, self.actor)
        hard_sync(self.target_critic, self.critic)
        hard_sync(self.target_critic_1, self.critic_1)

        self.global_step = 0

    def extra_repr(self) -> str:
        return f'gamma={self.gamma}, tau={self.tau}, target_noise_scale={self.target_noise_scale}, actor_update_steps={self.actor_update_steps}'

    def forward(self, observation, deterministic):
        #constructed_action, action, allocations, budgets
        constructed_action = self.actor(observation)
        return constructed_action, torch.nn.functional.normalize(constructed_action, dim=1, p=1), torch.ones((observation.shape[0], 1)), torch.sum(constructed_action, dim=1, keepdim=True)

    def _update(self, observation, allocations, budgets, action, reward, next_observation, done, truncateds):
        with torch.no_grad():
            next_action = self.target_actor(next_observation)
            noise = torch.normal(0, self.target_noise_scale, next_action.shape)
            next_action += noise

            target_q = torch.min(
                self.target_critic(next_observation, next_action),
                self.target_critic_1(next_observation, next_action)
            )
            target_q = reward + self.gamma * (1 - done) * target_q

        q = self.critic(observation, action * budgets)
        q_1 = self.critic_1(observation, action * budgets)

        critic_loss = torch.nn.functional.mse_loss(q, target_q)
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        critic_loss_1 = torch.nn.functional.mse_loss(q_1, target_q)
        self.critic_1.optimizer.zero_grad()
        critic_loss_1.backward()
        self.critic_1.optimizer.step()

        soft_sync(self.target_critic, self.critic, self.tau)
        soft_sync(self.target_critic_1, self.critic_1, self.tau)

        stats = {
            'loss/critic_loss': critic_loss.item(),
            'loss/critic_loss_1': critic_loss_1.item(),
        }

        self.global_step += 1
        if self.global_step % self.actor_update_steps == 0:
            actor_loss = -self.critic(observation, self.actor(observation)).mean()
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            soft_sync(self.target_actor, self.actor, self.tau)

            stats['loss/actor_loss'] = actor_loss.item()

        return stats

