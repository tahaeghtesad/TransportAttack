import numpy as np
import torch

from models import AllocatorInterface, DecayingNoiseInterface, CustomModule, EpsilonInterface
from util.torch.math import r2_score, explained_variance
from util.torch.misc import hard_sync, soft_sync
from util.torch.rl import GeneralizedAdvantageEstimation


class VCritic(CustomModule):

    def __init__(self, n_components, n_features, lr) -> None:
        super().__init__(name='VCritic')

        self.n_components = n_components
        self.n_features = n_features

        self.state_extractor = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.n_components * self.n_features, 64),
            # torch.nn.LayerNorm([64]),
            torch.nn.ReLU(),
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(64 + 1, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, aggregated_state, budget):

        return self.model(
            torch.cat((
                self.state_extractor(aggregated_state),
                budget
            ), dim=1)
        )

    def extra_repr(self) -> str:
        return f'n_components={self.n_components}, n_features={self.n_features}'


class QCritic(CustomModule):

    def __init__(self, n_components, n_features, lr) -> None:
        super().__init__(name='QCritic')

        self.n_components = n_components
        self.n_features = n_features

        self.state_extractor = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.n_components * self.n_features, 64),
            # torch.nn.LayerNorm([64]),
            torch.nn.ReLU(),
        )

        self.action_extractor = torch.nn.Sequential(
            torch.nn.Linear(self.n_components, 64),
            # torch.nn.LayerNorm([64]),
            torch.nn.ReLU(),
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(64 + 64 + 1, 1),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, aggregated_state, allocation, budget):

        return self.model(
            torch.cat((
                self.state_extractor(aggregated_state),
                self.action_extractor(allocation),
                budget
            ), dim=1)
        )

    def extra_repr(self) -> str:
        return f'n_components={self.n_components}, n_features={self.n_features}'


class StochasticActor(CustomModule):

    def __init__(self, n_components, n_features, max_concentration) -> None:
        super().__init__(name='StochasticActor')

        self.n_components = n_components
        self.n_features = n_features
        self.max_concentration = max_concentration

        self.state_extractor = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.n_components * self.n_features, 64),
            torch.nn.ReLU()
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(64 + 1, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.n_components),
            torch.nn.Sigmoid(),
            # torch.nn.Softmax(dim=1),
        )

    def forward(self, aggregated_state, budgets, deterministic):

        dist, _ = self.get_distribution(aggregated_state, budgets)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy()

    def get_distribution(self, aggregated_states, budgets):
        logits = self.get_logits(aggregated_states, budgets)
        # logits = torch.clamp(logits, 1e-4, 1e6)
        dist = torch.distributions.Dirichlet(concentration=logits * (self.max_concentration - 1) + 1)
        return dist, logits

    def get_logits(self, aggregated_states, budgets):
        return self.model(
            torch.cat((
                self.state_extractor(aggregated_states),
                budgets
            ), dim=1)
        )

    def extra_repr(self) -> str:
        return f'n_components={self.n_components}, n_features={self.n_features}, distribution={torch.distributions.Dirichlet.__class__.__name__}, max_concentration={self.max_concentration}'


class DeterministicActor(CustomModule):
    def __init__(self, n_components, n_features, lr) -> None:
        super().__init__('StochasticActor')

        self.n_components = n_components
        self.n_features = n_features

        self.state_extractor = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.n_components * self.n_features, 64),
            # torch.nn.LayerNorm([64]),
            torch.nn.ReLU(),
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(64 + 1, self.n_components),
            torch.nn.ReLU(),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, aggregated_state, budgets):
        logits = self.model(
            torch.cat((
                self.state_extractor(aggregated_state),
                budgets,
            ), dim=1)
        )
        return torch.nn.functional.normalize(logits, dim=1, p=1)

    def extra_repr(self) -> str:
        return f'n_components={self.n_components}, n_features={self.n_features}'


class DSPGAllocator(AllocatorInterface):

    def __init__(self, n_components, n_features, critic_lr, actor_lr, tau, gamma):
        super().__init__(name='DSPGBudgetAllocator')
        self.n_components = n_components
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.tau = tau
        self.gamma = gamma

        self.value = VCritic(n_components, n_features, critic_lr)
        self.actor = StochasticActor(n_components, n_features)

        self.target_value = VCritic(n_components, n_features, critic_lr)
        self.target_actor = StochasticActor(n_components, n_features)

        hard_sync(self.target_value, self.value)
        hard_sync(self.target_actor, self.actor)

    def forward(self, aggregated_state, budgets, deterministic):
        with torch.no_grad():
            action = self.actor.forward(aggregated_state, budgets, deterministic)
        return action

    def update(self, aggregated_states, allocations, budgets, rewards, next_aggregated_states, next_budgets, dones, truncateds):
        # updating actor

        distributions, logits = self.actor.get_distribution(aggregated_states, budgets)
        log_prob = distributions.log_prob(allocations)
        entropy = distributions.entropy()
        values = self.value.forward(aggregated_states, budgets)

        actor_loss = - (log_prob * (values - values.mean().detach()) + entropy).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # updating critic

        with torch.no_grad():
            next_values = self.target_value.forward(next_aggregated_states, next_budgets)
            target_values = rewards + (1 - dones) * self.gamma * next_values

        values = self.value.forward(aggregated_states, budgets)
        critic_loss = torch.nn.functional.mse_loss(values, target_values)

        self.value.optimizer.zero_grad()
        critic_loss.backward()
        self.value.optimizer.step()

        # updating target networks
        soft_sync(self.target_value, self.value, self.tau)
        soft_sync(self.target_actor, self.actor, self.tau)

        return {
            'allocator/v_loss': critic_loss.detach().cpu().item(),
            'allocator/v_min': values.min().detach().cpu().item(),
            'allocator/v_max': values.max().detach().cpu().item(),
            'allocator/v_mean': values.mean().detach().cpu().item(),
            'allocator/a_val': -actor_loss.detach().cpu().item(),
            'allocator/entropy': entropy.mean().detach().cpu().item(),
            'allocator/r2': max(r2_score(target_values, values).detach().cpu().item(), -1),
            'allocator/explained_variance': max(explained_variance(target_values, values).detach().cpu().item(), -1),
        }


class DDPGAllocator(AllocatorInterface):
    def __init__(self, edge_component_mapping, n_features, critic_lr, actor_lr, tau, gamma, noise: DecayingNoiseInterface, epsilon: EpsilonInterface):
        super().__init__(name='DDPGBudgetAllocator')
        self.edge_component_mapping = edge_component_mapping
        self.n_components = len(edge_component_mapping)
        self.n_features = n_features

        self.critic_lr = critic_lr
        self.actor_lr = actor_lr

        self.critic = QCritic(self.n_components, n_features, critic_lr)
        self.actor = DeterministicActor(self.n_components, n_features, actor_lr)
        self.target_critic = QCritic(self.n_components, n_features, critic_lr)
        self.target_actor = DeterministicActor(self.n_components, n_features, actor_lr)

        self.noise = noise
        self.epsilon = epsilon

        self.tau = tau
        self.gamma = gamma

        hard_sync(self.target_critic, self.critic)
        hard_sync(self.target_actor, self.actor)

    def forward(self, aggregated_states, budgets, deterministic):
        with torch.no_grad():
            actions = self.actor.forward(aggregated_states, budgets)
        if not deterministic:
            if self.epsilon():
                actions = torch.nn.functional.normalize(torch.rand(actions.shape), dim=1, p=1)
            actions = torch.nn.functional.normalize(torch.maximum(self.noise.forward(actions.shape) + actions, torch.tensor(0, device=self.device)), dim=1, p=1)
        return actions

    def update(self, aggregated_states, allocations, budgets, rewards, next_aggregated_states, next_budgets, dones, truncateds):

        # Update critic
        with torch.no_grad():
            next_allocations = self.target_actor.forward(next_aggregated_states, next_budgets)
            next_q_values = self.target_critic.forward(next_aggregated_states, next_allocations, next_budgets)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        q_values = self.critic.forward(aggregated_states, allocations, budgets)
        critic_loss = torch.nn.functional.mse_loss(q_values, target_q_values)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Update actor
        current_action = self.actor.forward(aggregated_states, budgets)
        current_q_values = self.critic.forward(aggregated_states, current_action, budgets)

        actor_loss = - current_q_values.mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Update target networks
        soft_sync(self.target_critic, self.critic, self.tau)
        soft_sync(self.target_actor, self.actor, self.tau)

        return {
            'allocator/q_loss': critic_loss.detach().cpu().item(),
            'allocator/q_min': q_values.min().detach().cpu().item(),
            'allocator/q_max': q_values.max().detach().cpu().item(),
            'allocator/q_mean': q_values.mean().detach().cpu().item(),
            'allocator/a_val': -actor_loss.detach().cpu().item(),
            'allocator/r2': max(r2_score(target_q_values, q_values).detach().cpu().item(), -1),
            'allocator/explained_variance': max(explained_variance(target_q_values, q_values).detach().cpu().item(), -1),
            'allocator/noise': self.noise.get_current_noise().detach().cpu().item(),
            'allocator/epsilon': self.epsilon.get_current_epsilon().detach().cpu().item(),
        }


class PPOAllocator(AllocatorInterface):
    def __init__(self, edge_component_mapping, n_features, actor_lr, critic_lr, gamma, lam, epsilon, policy_grad_clip, value_coeff, entropy_coeff, n_updates, batch_size, max_concentration, clip_range_vf):
        super().__init__(name='PPOBudgetAllocator')
        self.lam = lam
        self.epsilon = epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.n_updates = n_updates
        self.policy_grad_clip = policy_grad_clip
        self.batch_size = batch_size
        self.clip_range_vf = clip_range_vf

        self.edge_component_mapping = edge_component_mapping
        self.n_components = len(edge_component_mapping)
        self.n_features = n_features

        self.value = VCritic(self.n_components, n_features, critic_lr)
        self.actor = StochasticActor(self.n_components, n_features, max_concentration)

        self.gae = GeneralizedAdvantageEstimation(gamma, lam)
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': actor_lr},
            {'params': self.value.parameters(), 'lr': critic_lr},
        ])

    def extra_repr(self) -> str:
        return f'actor_lr={self.actor_lr}, critic_lr={self.critic_lr}, policy_grad_clip={self.policy_grad_clip}, gamma={self.gamma}, lam={self.lam}, epsilon={self.epsilon}, value_coeff={self.value_coeff}, entropy_coeff={self.entropy_coeff}, n_updates={self.n_updates}, clip_range_vf={self.clip_range_vf}'

    def forward(self, aggregated_states, budgets, deterministic):
        with torch.no_grad():
            action, log_prob, entropy = self.actor.forward(aggregated_states, budgets, deterministic)
        return action

    def update(self, aggregated_states, allocations, budgets, rewards, next_aggregated_states, next_budgets, dones, truncateds):

        data_points = aggregated_states.shape[0]
        assert data_points % self.batch_size == 0, f'batch_size={self.batch_size} does not divide buffer_size={data_points}'

        log_prob_diff_history = []

        with torch.no_grad():
            old_distribution, old_logits = self.actor.get_distribution(aggregated_states, budgets)
            old_values = self.value.forward(aggregated_states, budgets)
            old_log_prob = torch.unsqueeze(old_distribution.log_prob(allocations), dim=1)
            log_prob_diff_history.append(old_log_prob.squeeze().detach().cpu().numpy())

        value_loss_history = []
        loss_history = []
        actor_loss_history = []
        r2_history = []
        min_advantage_history = []
        max_advantage_history = []

        for epoch in range(self.n_updates):

            # calculating gae
            with torch.no_grad():
                old_values = self.value.forward(aggregated_states, budgets)
                advantages = self.gae.forward(old_values, rewards, dones, truncateds)
                returns = advantages + old_values

                min_advantage_history.append(advantages.min().detach().cpu().item())
                max_advantage_history.append(advantages.max().detach().cpu().item())

            permutations = torch.randperm(data_points)

            for batch in range(data_points // self.batch_size):

                indices = permutations[batch * self.batch_size: (batch + 1) * self.batch_size]

                # calculating value loss
                values = self.value.forward(aggregated_states[indices], budgets[indices])
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = old_values[indices] + torch.clamp(
                        values - old_values[indices], -self.clip_range_vf, self.clip_range_vf
                    )
                value_loss = torch.nn.functional.mse_loss(values_pred, returns[indices])
                r2_history.append(r2_score(returns[indices], values).detach().cpu().item())

                # with torch.no_grad():
                #     next_values = self.value.forward(next_aggregated_states[indices], next_budgets[indices])
                #     target_values = rewards[indices] + (1 - dones[indices]) * self.gamma * next_values
                #
                # values = self.value.forward(aggregated_states[indices], budgets[indices])
                # value_loss = torch.nn.functional.mse_loss(values, target_values)
                # r2_history.append(r2_score(target_values, values).detach().cpu().item())

                value_loss_history.append(value_loss.detach().cpu().item())

                # calculating actor loss
                batch_advantages = (advantages[indices] - advantages[indices].mean()) / (advantages[indices].std())
                # batch_advantages = advantages[indices]
                distribution, logits = self.actor.get_distribution(aggregated_states[indices], budgets[indices])
                new_log_prob = torch.unsqueeze(distribution.log_prob(allocations[indices]), dim=1)
                ratio = torch.exp(new_log_prob - old_log_prob[indices])
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                actor_loss = torch.min(surr1, surr2).mean()
                actor_loss_history.append(actor_loss.detach().cpu().item())

                # calculating kl divergence
                # kl_divergence = torch.distributions.kl.kl_divergence(old_distribution, distribution).mean()

                # calculating entropy
                entropy = distribution.entropy().mean()

                # calculating total loss
                loss = - actor_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy
                loss_history.append(loss.detach().cpu().item())

                # update network
                self.optimizer.zero_grad()
                loss.backward()
                if self.policy_grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.policy_grad_clip)
                self.optimizer.step()

            with torch.no_grad():
                distribution, logits = self.actor.get_distribution(aggregated_states, budgets)
                log_probs = torch.unsqueeze(distribution.log_prob(allocations), dim=1)
                log_prob_diff_history.append(log_probs.squeeze().detach().cpu().numpy())

        # for the sake of logging
        with torch.no_grad():
            distribution, logits = self.actor.get_distribution(aggregated_states, budgets)
            kl_divergence = torch.distributions.kl.kl_divergence(old_distribution, distribution)
            new_log_prob = torch.unsqueeze(distribution.log_prob(allocations), dim=1)
            ratio = torch.exp(new_log_prob - old_log_prob)
            entropy = torch.unsqueeze(distribution.entropy(), dim=1)
            concentration = distribution.concentration

            # if kl_divergence > self.kl_target * 1.5:
            #     self.kl_coeff *= 2
            # elif kl_divergence < self.kl_target / 1.5:
            #     self.kl_coeff /= 2
            #
            # self.kl_coeff = torch.max(torch.tensor(1e-2, device=self.device), self.kl_coeff)

            # if kl_divergence < 1e-7:
            #     raise Exception(f'kl_divergence is too small... Policy not updated... exiting... {kl_divergence:.8f}|{ratio.max():.8f}|{ratio.min():.8f}')

        view = np.array(log_prob_diff_history)
        view = (view - view.mean(axis=0)) / view.std(axis=0)
        # print(view)

        return {
            'allocator/v_loss': np.mean(value_loss_history),
            'allocator/kl_divergence_mean': kl_divergence.mean().detach().cpu().item(),
            'allocator/kl_divergence_max': kl_divergence.max().detach().cpu().item(),
            'allocator/entropy_mean': entropy.mean().detach().cpu().item(),
            'allocator/concentration_mean': concentration.mean().detach().cpu().item(),
            'allocator/concentration_min': concentration.min().detach().cpu().item(),
            'allocator/concentration_max': concentration.max().detach().cpu().item(),
            # 'allocator/loss': np.mean(loss_history),
            'allocator/ratio_mean': ratio.mean().detach().cpu().item(),
            'allocator/ratio_max': ratio.max().detach().cpu().item(),
            'allocator/ratio_min': ratio.min().detach().cpu().item(),
            'allocator/actor_loss': np.mean(actor_loss_history),
            'allocator/r2': np.maximum(np.mean(r2_history), -1),
            'allocator/log_probs': new_log_prob.detach().cpu().numpy().tolist(),
            'allocator/advantages': advantages.flatten().detach().cpu().numpy().tolist(),
        }
