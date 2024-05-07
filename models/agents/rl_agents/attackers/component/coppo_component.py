import numpy as np
import torch

from models import CustomModule
from models.agents.rl_agents.attackers.component import ComponentInterface
from util.torch.rl import GeneralizedAdvantageEstimation


class VCritic(CustomModule):
    def __init__(self, adj, n_features, n_edges, lr):
        super().__init__('Critic')

        self.n_features = n_features
        self.n_edges = n_edges
        self.lr = lr
        self.adj = adj

        self.state_extractor = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(n_features * n_edges, 256),
            torch.nn.Softplus(),
        )

        self.budget_extractor = torch.nn.Sequential(
            torch.nn.Linear(1, 256),
            torch.nn.Softplus(),
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.Softplus(),
            torch.nn.Linear(256, 1),
        )

    def extra_repr(self) -> str:
        return f'n_features={self.n_features}, n_edges={self.n_edges}, lr={self.lr}'

    def forward(self, observation, budget):
        return self.model(
            torch.cat((self.state_extractor(observation), self.budget_extractor(budget)), dim=1)
        )


class StochasticActor(CustomModule):

    def __init__(self, adj, n_features, n_edges, lr):
        super().__init__('Actor')

        self.n_features = n_features
        self.n_edges = n_edges
        self.lr = lr
        self.adj = adj

        self.state_extractor = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(n_features * n_edges, 256),
            torch.nn.Softplus(),
        )

        self.budget_extractor = torch.nn.Sequential(
            torch.nn.Linear(1, 256),
            torch.nn.Softplus(),
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.Softplus(),
            torch.nn.Linear(256, n_edges),
            torch.nn.Softplus(),
        )

    def extra_repr(self) -> str:
        return f'n_features={self.n_features}, n_edges={self.n_edges}, lr={self.lr}'

    def get_distribution(self, logits):
        return torch.distributions.Dirichlet(concentration=logits + 1)

    def get_logits(self, observation, budget):
        return self.model(
            torch.cat((self.state_extractor(observation), self.budget_extractor(budget)), dim=1)
        )

    def forward(self, observation, budget, deterministic):
        logits = self.get_logits(observation, budget)
        distribution = self.get_distribution(logits)
        action = distribution.sample() if not deterministic else distribution.mean
        return action


class CoPPOComponent(ComponentInterface):

    def __init__(self, adj, edge_component_mapping, n_features, actor_lr, critic_lr, gamma, lam, epsilon, joint_epsilon,
                 policy_grad_clip, value_coeff, entropy_coeff, n_updates, batch_size, clip_range_vf,
                 normalize_advantages) -> None:
        super().__init__('CoPPOComponent')
        self.edge_component_mapping = edge_component_mapping
        self.n_components = len(edge_component_mapping)

        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.joint_epsilon = joint_epsilon
        self.policy_grad_clip = policy_grad_clip
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.n_updates = n_updates
        self.batch_size = batch_size
        self.clip_range_vf = clip_range_vf
        self.normalize_advantages = normalize_advantages

        self.actors = torch.nn.ModuleList([
            StochasticActor(adj, n_features, len(c), actor_lr)
            for c in edge_component_mapping
        ])

        self.values = torch.nn.ModuleList([
            VCritic(adj, n_features, len(c), critic_lr)
            for c in edge_component_mapping
        ])

        self.assignments = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(n_features * sum([len(c) for c in edge_component_mapping]), self.n_components),
            torch.nn.Softmax(dim=1),
        )

        self.optimizer = torch.optim.Adam([
            {'params': self.actors.parameters(), 'lr': actor_lr},
            {'params': self.values.parameters(), 'lr': critic_lr},
            {'params': self.assignments.parameters(), 'lr': actor_lr},
        ])

        self.gae = GeneralizedAdvantageEstimation(gamma=self.gamma, lam=self.lam)

        self.one_hot_mask = torch.nn.functional.one_hot(torch.arange(self.n_components, device=self.device),
                                                        num_classes=self.n_components).bool()

    def extra_repr(self) -> str:
        f'edge_component_mapping={self.edge_component_mapping}, n_components={self.n_components}, gamma={self.gamma}, lam={self.lam}, epsilon={self.epsilon}, joint_epsilon={self.joint_epsilon}, policy_grad_clip={self.policy_grad_clip}, value_coeff={self.value_coeff}, entropy_coeff={self.entropy_coeff}, n_updates={self.n_updates}, batch_size={self.batch_size}, clip_range_vf={self.clip_range_vf}, normalize_advantages={self.normalize_advantages}'

    def __forward_value(self, observation, budget):

        values = torch.empty(
            (observation.shape[0], 1),
            device=self.device)

        assignments = self.assignments.forward(observation)

        for c in range(self.n_components):
            values += self.values[c].forward(observation[:, self.edge_component_mapping[c], :],
                                                    budget) * assignments[:, [c]]

        return values

    def __forward_actor(self, observation, budget, allocations, deterministic):

        actions = torch.empty(
            (observation.shape[0], sum([len(c) for c in self.edge_component_mapping])),
            device=self.device)

        for c in range(self.n_components):
            actions[:, self.edge_component_mapping[c]] = self.actors[c].forward(
                observation[:, self.edge_component_mapping[c], :], budget * allocations[:, [c]], deterministic)

        return actions

    def __get_logits(self, observation, budget, allocations):
        old_logits = torch.empty(
            (observation.shape[0], sum([len(c) for c in self.edge_component_mapping])),
            device=self.device)

        for c in range(self.n_components):
            old_logits[:, self.edge_component_mapping[c]] = self.actors[c].get_logits(
                observation[:, self.edge_component_mapping[c], :], budget * allocations[:, [c]])

        return old_logits

    def __get_distributions(self, old_logits):
        distributions = []
        for c in range(self.n_components):
            distributions.append(self.actors[c].get_distribution(old_logits[:, self.edge_component_mapping[c]]))
        return distributions

    def __get_log_probs(self, distributions, action):
        log_probs = torch.empty(
            (action.shape[0], self.n_components),
            device=self.device)

        for c in range(self.n_components):
            log_probs[:, c] = distributions[c].log_prob(action[:, self.edge_component_mapping[c]])

        return log_probs

    def forward(self, states, budgets, allocations, deterministic):
        return self.__forward_actor(states, budgets, allocations, deterministic)

    def update(self, states, actions, budgets, allocations, next_states, next_budgets, next_allocations, rewards, dones,
               truncateds):
        n_samples = states.shape[0]
        assert n_samples % self.batch_size == 0, 'Batch size must be a multiple of n_samples'

        with torch.no_grad():
            old_values = self.__forward_value(states, budgets)
            old_next_values = self.__forward_value(next_states, next_budgets)
            advantages = self.gae.forward(old_values, torch.sum(rewards, dim=1, keepdim=True), dones, truncateds, old_next_values)
            returns = advantages + old_values
            old_logits = self.__get_logits(states, budgets, allocations)
            old_distribution = self.__get_distributions(old_logits)
            old_log_prob = self.__get_log_probs(old_distribution, actions)

        for epoch in range(self.n_updates):
            permutations = torch.randperm(n_samples)
            for batch in range(n_samples // self.batch_size):
                indices = permutations[batch * self.batch_size: (batch + 1) * self.batch_size]

                values = self.__forward_value(states[indices], budgets[indices])

                value_loss = torch.nn.functional.mse_loss(values, returns[indices])

                actor_loss = torch.zeros(1, device=self.device)
                logits = self.__get_logits(states[indices], budgets[indices], allocations[indices])
                distribution = self.__get_distributions(logits)
                new_log_prob = self.__get_log_probs(distribution, actions[indices])

                assignments = self.assignments.forward(states[indices])

                for c in range(self.n_components):
                    mask = torch.logical_not(self.one_hot_mask[c])
                    g_r = torch.sum(new_log_prob[:, mask] - old_log_prob[indices][:, mask], dim=1, keepdim=True)
                    g_r = torch.clamp(g_r, np.log(1 - self.joint_epsilon), np.log(1 + self.joint_epsilon))

                    log_ratio = new_log_prob[:, [c]] - old_log_prob[indices][:, [c]] + g_r

                    batch_adv = assignments[:, c] * advantages[indices]
                    if self.normalize_advantages:
                        batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-8)

                    surr1 = log_ratio * batch_adv
                    surr2 = torch.clamp(log_ratio, np.log(1 - self.epsilon), np.log(1 + self.epsilon)) * batch_adv
                    actor_loss += -torch.min(surr1, surr2).mean()

                loss = actor_loss + self.value_coeff * value_loss + self.entropy_coeff * logits.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        with torch.no_grad():
            logits = self.__get_logits(states, budgets, allocations)
            distribution = self.__get_distributions(logits)
            log_prob = self.__get_log_probs(distribution, actions)
            entropy = sum([d.entropy().mean() for d in distribution]) / self.n_components

        return {
            'attacker/actor_loss': actor_loss.detach().cpu().numpy().item(),
            'attacker/value_loss': value_loss.detach().cpu().numpy().item(),
            'attacker/advantages': advantages.detach().cpu().numpy().mean(),
            'attacker/max_concentration': old_logits.detach().cpu().numpy().max(),
            'attacker/mean_concentration': old_logits.detach().cpu().numpy().mean(),
            'attacker/mean_entropy': entropy.detach().cpu().numpy().mean(),
            'attacker/max_log_prob': torch.sum(log_prob, dim=1).detach().cpu().numpy().max(),
            'attacker/min_log_prob': torch.sum(log_prob, dim=1).detach().cpu().numpy().min(),
            'attacker/max_log_prob_change': (
                        torch.sum(log_prob, dim=1) - torch.sum(old_log_prob, dim=1)).detach().cpu().numpy().max(),
            'attacker/min_log_prob_change': (
                        torch.sum(log_prob, dim=1) - torch.sum(old_log_prob, dim=1)).detach().cpu().numpy().min(),
            'attacker/decomposition': torch.mean(self.assignments.forward(states), dim=0).detach().cpu().numpy()
        }
