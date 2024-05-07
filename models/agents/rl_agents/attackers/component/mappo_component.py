import numpy as np
import torch

from models import CustomModule
from models.agents.rl_agents.attackers.component import ComponentInterface
from util.torch.math import r2_score
from util.torch.rl import GeneralizedAdvantageEstimation


class VCritic(CustomModule):
    def __init__(self, name, n_edges, n_features) -> None:
        super().__init__(name)
        self.n_edges = n_edges
        self.n_features = n_features

        self.state_extractor = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(n_edges * n_features, 64),
            torch.nn.Softplus(),
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(64 + 2, 128),
            torch.nn.Softplus(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, observation, budget, allocation):
        return self.model(
            torch.cat(
                (self.state_extractor(observation), budget, allocation), dim=1
            )
        )

    def extra_repr(self) -> str:
        return f'n_edges={self.n_edges}, n_features={self.n_features}'


class StochasticActor(CustomModule):

    def __init__(self, name, n_edges, n_features) -> None:
        super().__init__(name)
        self.n_edges = n_edges
        self.n_features = n_features

        self.state_extractor = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(n_edges * n_features, 64),
            torch.nn.Softplus(),
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(64 + 2, 128),
            torch.nn.Softplus(),
            torch.nn.Linear(128, n_edges),
            torch.nn.Softplus()
        )

    def forward(self, observation, budget, allocation, deterministic):
        distribution = self.get_distribution(observation, budget, allocation)
        if deterministic:
            return distribution.mean
        else:
            return distribution.sample()

    def get_logits(self, observation, budget, allocation):
        return self.model(
            torch.cat(
                (self.state_extractor(observation), allocation, budget), dim=1
            )
        )

    def get_distribution(self, observation, budget, allocation):
        concentration = self.get_logits(observation, budget, allocation) + 1  # * (self.max_concentration - 1) + 1
        return torch.distributions.Dirichlet(concentration)

    def extra_repr(self) -> str:
        return f'n_edges={self.n_edges}, n_features={self.n_features}'


class MAPPOComponentAgent(CustomModule):
    def __init__(self, index, n_edges, n_features, actor_lr, critic_lr, gamma, lam, epsilon, policy_grad_clip, value_coeff, entropy_coeff, n_updates, batch_size, clip_range_vf, normalize_advantages) -> None:
        super().__init__(f'PPOComponentAgent-{index}')
        self.index = index
        self.n_edges = n_edges
        self.n_features = n_features
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.policy_grad_clip = policy_grad_clip
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.n_updates = n_updates
        self.batch_size = batch_size
        self.clip_range_vf = clip_range_vf
        self.normalize_advantages = normalize_advantages

        self.value = VCritic(f'Value-{index}', n_edges, n_features)
        self.actor = StochasticActor(f'Actor-{index}', n_edges, n_features)

        self.gae = GeneralizedAdvantageEstimation(self.gamma, self.lam)
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': self.actor_lr},
            {'params': self.value.parameters(), 'lr': self.critic_lr},
        ])

    def extra_repr(self) -> str:
        return f'n_features={self.n_features}, actor_lr={self.actor_lr}, critic_lr={self.critic_lr}, gamma={self.gamma}, lam={self.lam}, epsilon={self.epsilon}, policy_grad_clip={self.policy_grad_clip}, value_coeff={self.value_coeff}, entropy_coeff={self.entropy_coeff}, n_updates={self.n_updates}, batch_size={self.batch_size}, clip_range_vf={self.clip_range_vf}'

    def forward(self, states, budgets, allocations, deterministic):
        return self.actor.forward(states, budgets, allocations, deterministic=deterministic)

    def update(self, states, actions, budgets, allocations, rewards, dones, truncateds):
        data_points = states.shape[0]
        assert data_points % self.batch_size == 0, f'batch_size={self.batch_size} does not divide buffer_size={data_points}'

        with torch.no_grad():
            old_values = self.value.forward(states, budgets, allocations)
            advantages = self.gae.forward(old_values, rewards, dones, truncateds)
            returns = advantages + old_values
            old_distribution = self.actor.get_distribution(states, budgets, allocations)
            old_log_prob = torch.unsqueeze(old_distribution.log_prob(actions), dim=1)

        value_loss_history = []
        loss_history = []
        actor_loss_history = []
        r2_history = []

        for epoch in range(self.n_updates):

            permutations = torch.randperm(data_points)

            for batch in range(data_points // self.batch_size):

                indices = permutations[batch * self.batch_size: (batch + 1) * self.batch_size]

                # calculating value loss
                values = self.value.forward(states[indices], budgets[indices], allocations[indices])
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = old_values[indices] + torch.clamp(
                        values - old_values[indices], -self.clip_range_vf, self.clip_range_vf
                    )
                value_loss = torch.nn.functional.mse_loss(values_pred, returns[indices])
                r2_history.append(r2_score(returns[indices], values).detach().cpu().item())
                value_loss_history.append(value_loss.detach().cpu().item())

                # calculating actor loss
                if self.normalize_advantages:
                    batch_advantages = (advantages[indices] - advantages[indices].mean()) / (
                        advantages[indices].std() + 1e-8)
                else:
                    batch_advantages = advantages[indices]

                distribution = self.actor.get_distribution(states[indices], budgets[indices],
                                                                           allocations[indices])
                new_log_prob = torch.unsqueeze(distribution.log_prob(actions[indices]), dim=1)
                log_ratio = new_log_prob - old_log_prob[indices]
                surr1 = log_ratio * batch_advantages
                surr2 = torch.clamp(log_ratio, np.log(1.0 - self.epsilon),
                                    np.log(1.0 + self.epsilon)) * batch_advantages
                actor_loss = - torch.min(surr1, surr2).mean()
                actor_loss_history.append(actor_loss.detach().cpu().item())

                # calculating kl divergence
                # kl_divergence = torch.distributions.kl.kl_divergence(old_distribution, distribution).mean()

                # calculating entropy
                entropy = distribution.entropy().mean()

                # calculating total loss
                loss = + actor_loss + self.value_coeff * value_loss + self.entropy_coeff * distribution.concentration.mean()
                loss_history.append(loss.detach().cpu().item())

                # update network
                self.optimizer.zero_grad()
                loss.backward()
                if self.policy_grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.policy_grad_clip)
                self.optimizer.step()

        # for the sake of logging
        with torch.no_grad():
            distributions = self.actor.get_distribution(states, budgets, allocations)
            kl_divergence = torch.distributions.kl.kl_divergence(old_distribution, distributions).mean()
            new_log_prob = distributions.log_prob(actions)
            # ratio = torch.exp(new_log_prob - old_log_prob)
            ratio = new_log_prob - old_log_prob
            entropy = distributions.entropy().mean()
            concentration = distributions.concentration

        return {
            f'component/{self.index}/v_loss': np.mean(value_loss_history),
            f'component/{self.index}/entropy_mean': entropy.mean().detach().cpu().item(),
            f'component/{self.index}/concentration_mean': concentration.mean().detach().cpu().item(),
            f'component/{self.index}/concentration_max': concentration.max().detach().cpu().item(),
            f'component/{self.index}/actor_loss': np.mean(actor_loss_history),
            f'component/{self.index}/advantages': advantages.mean().detach().cpu().item(),
        }


class MAPPOComponent(ComponentInterface):

    def __init__(self, edge_component_mapping, n_features, actor_lr, critic_lr, gamma, lam, epsilon, policy_grad_clip, value_coeff, entropy_coeff, n_updates, batch_size, clip_range_vf, normalize_advantages) -> None:
        super().__init__('MAPPOComponent')
        self.edge_component_mapping = edge_component_mapping
        self.n_components = len(edge_component_mapping)

        self.agents = torch.nn.ModuleList([
            MAPPOComponentAgent(
                i, len(edge_component_mapping[i]), n_features, actor_lr, critic_lr, gamma, lam, epsilon, policy_grad_clip,
                value_coeff, entropy_coeff, n_updates, batch_size, clip_range_vf, normalize_advantages
            ) for i in range(self.n_components)
        ])

    def extra_repr(self) -> str:
        return f'n_components={self.n_components}'

    def forward(self, states, budgets, allocations, deterministic):
        actions = torch.empty(
            (states.shape[0], sum([len(c) for c in self.edge_component_mapping])),
            device=self.device)

        for c in range(self.n_components):
            component_action = self.agents[c].forward(
                states[:, self.edge_component_mapping[c], :],
                budgets,
                allocations[:, [c]],
                deterministic=deterministic
            )
            actions[:, self.edge_component_mapping[c]] = component_action

        return actions

    def update(self, states, actions, budgets, allocations, next_states, next_budgets, next_allocations, rewards, dones, truncateds):
        stats = {}
        for c in range(self.n_components):
            stats |= self.agents[c].update(
                states[:, self.edge_component_mapping[c], :],
                actions[:, self.edge_component_mapping[c]],
                budgets * allocations[:, [c]],
                allocations[:, [c]],
                rewards[:, [c]],
                dones,
                truncateds
            )

        return stats
