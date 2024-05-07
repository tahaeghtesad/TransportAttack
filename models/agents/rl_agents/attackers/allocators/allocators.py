import numpy as np
import torch

from models import CustomModule
from models.agents.rl_agents.attackers.allocators import AllocatorInterface, NoBudgetAllocatorInterface
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
            torch.nn.Linear(64 + 1, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
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
            torch.nn.Linear(self.n_components * self.n_features + 1, 256),
            torch.nn.ReLU(),
        )

        self.action_extractor = torch.nn.Sequential(
            torch.nn.Linear(self.n_components, 256),
            torch.nn.ReLU(),
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(256 + 256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, aggregated_state, allocation, budget):

        return self.model(
            torch.cat((
                self.state_extractor(
                    torch.cat((
                        torch.flatten(aggregated_state, start_dim=1),
                        budget
                    ), dim=1)
                ),
                self.action_extractor(allocation),
            ), dim=1)
        )

    def extra_repr(self) -> str:
        return f'n_components={self.n_components}, n_features={self.n_features}'


class StochasticActor(CustomModule):

    def __init__(self, n_components, n_features) -> None:
        super().__init__(name='StochasticActor')

        self.n_components = n_components
        self.n_features = n_features

        self.state_extractor = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.n_components * self.n_features, 64),
            torch.nn.Softplus()
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(64 + 1, 128),
            torch.nn.Softplus(),
            torch.nn.Linear(128, self.n_components),
            torch.nn.Softplus(),
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
        dist = torch.distributions.Dirichlet(concentration=logits + 1)
        return dist, logits

    def get_logits(self, aggregated_states, budgets):
        return self.model(
            torch.cat((
                self.state_extractor(aggregated_states),
                budgets
            ), dim=1)
        )

    def extra_repr(self) -> str:
        return f'n_components={self.n_components}, n_features={self.n_features}, distribution={torch.distributions.Dirichlet}'


class DeterministicActor(CustomModule):
    def __init__(self, n_components, n_features, lr) -> None:
        super().__init__('StochasticActor')

        self.n_components = n_components
        self.n_features = n_features

        self.state_extractor = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.n_components * self.n_features + 1, 512),
            torch.nn.ReLU(),
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, self.n_components),
            # torch.nn.Softmax(dim=1),
            # torch.nn.ReLU(),
            torch.nn.Softplus(),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, aggregated_state, budgets):
        logits = self.model(
            self.state_extractor(
                torch.cat((
                    torch.flatten(aggregated_state, start_dim=1),
                    budgets
                ), dim=1)
            ),
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
    def __init__(self, edge_component_mapping, n_features, critic_lr, actor_lr, tau, gamma):
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

        self.tau = tau
        self.gamma = gamma

        hard_sync(self.target_critic, self.critic)
        hard_sync(self.target_actor, self.actor)

    def forward(self, aggregated_states, budgets, deterministic):
        with torch.no_grad():
            actions = self.actor.forward(aggregated_states, budgets)
        if not deterministic:
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
        }


class PPOAllocator(AllocatorInterface):
    def __init__(self, edge_component_mapping, n_features, actor_lr, critic_lr, gamma, lam, epsilon, policy_grad_clip, value_coeff, entropy_coeff, n_updates, batch_size, clip_range_vf, kl_coeff, normalize_advantages):
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
        self.kl_coeff = kl_coeff
        self.normalize_advantages = normalize_advantages

        self.edge_component_mapping = edge_component_mapping
        self.n_components = len(edge_component_mapping)
        self.n_features = n_features

        self.value = VCritic(self.n_components, n_features, critic_lr)
        self.actor = StochasticActor(self.n_components, n_features)

        self.gae = GeneralizedAdvantageEstimation(gamma, lam)
        self.optimizer = torch.optim.RMSprop([
            {'params': self.actor.parameters(), 'lr': actor_lr},
            {'params': self.value.parameters(), 'lr': critic_lr},
        ])

    def extra_repr(self) -> str:
        return f'actor_lr={self.actor_lr}, critic_lr={self.critic_lr}, policy_grad_clip={self.policy_grad_clip}, gamma={self.gamma}, lam={self.lam}, epsilon={self.epsilon}, value_coeff={self.value_coeff}, entropy_coeff={self.entropy_coeff}, n_updates={self.n_updates}, clip_range_vf={self.clip_range_vf}, kl_coeff={self.kl_coeff}, normalize_advantages={self.normalize_advantages}'

    def forward(self, aggregated_states, budgets, deterministic):
        with torch.no_grad():
            action, log_prob, entropy = self.actor.forward(aggregated_states, budgets, deterministic)
        return action

    def update(self, aggregated_states, allocations, budgets, rewards, next_aggregated_states, next_budgets, dones, truncateds):

        data_points = aggregated_states.shape[0]
        assert data_points % self.batch_size == 0, f'batch_size={self.batch_size} does not divide buffer_size={data_points}'

        with torch.no_grad():
            old_values = self.value.forward(aggregated_states, budgets)
            advantages = self.gae.forward(old_values, rewards, dones, truncateds)
            returns = advantages + old_values
            old_distribution, old_logits = self.actor.get_distribution(aggregated_states, budgets)
            old_log_prob = torch.unsqueeze(old_distribution.log_prob(allocations), dim=1)

        value_loss_history = []
        loss_history = []
        actor_loss_history = []
        r2_history = []

        for epoch in range(self.n_updates):

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
                value_loss_history.append(value_loss.detach().cpu().item())

                # calculating actor loss
                if self.normalize_advantages:
                    batch_advantages = (advantages[indices] - advantages[indices].mean()) / (advantages[indices].std() + 1e-8)
                else:
                    batch_advantages = advantages[indices]

                distribution, logits = self.actor.get_distribution(aggregated_states[indices], budgets[indices])
                new_log_prob = torch.unsqueeze(distribution.log_prob(allocations[indices]), dim=1)
                log_ratio = new_log_prob - old_log_prob[indices]
                surr1 = log_ratio * batch_advantages
                surr2 = torch.clamp(log_ratio, np.log(1.0 - self.epsilon), np.log(1.0 + self.epsilon)) * batch_advantages
                actor_loss = - torch.min(surr1, surr2).mean()
                actor_loss_history.append(actor_loss.detach().cpu().item())

                # calculating kl divergence
                kl_divergence = torch.distributions.kl.kl_divergence(distribution, torch.distributions.Dirichlet(old_distribution.concentration[indices])).mean()

                # calculating entropy
                entropy = - distribution.entropy().mean()

                # calculating total loss
                loss = + actor_loss + self.value_coeff * value_loss + self.entropy_coeff * logits.mean()
                loss_history.append(loss.detach().cpu().item())

                # update network
                self.optimizer.zero_grad()
                loss.backward()
                if self.policy_grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.policy_grad_clip)
                self.optimizer.step()

        # for the sake of logging
        with torch.no_grad():
            values = self.value.forward(aggregated_states, budgets)
            distribution, logits = self.actor.get_distribution(aggregated_states, budgets)
            kl_divergence = torch.distributions.kl.kl_divergence(old_distribution, distribution)
            new_log_prob = torch.unsqueeze(distribution.log_prob(allocations), dim=1)
            log_ratio = new_log_prob - old_log_prob
            # ratio = torch.exp(log_ratio)
            entropy = torch.unsqueeze(distribution.entropy(), dim=1)
            concentration = distribution.concentration
            approx_kl_divergence = torch.exp(log_ratio) - 1 - log_ratio

        return {
            'allocator/v_loss': np.mean(value_loss_history),
            'allocator/entropy_mean': entropy.mean().detach().cpu().item(),
            'allocator/concentration_mean': concentration.mean().detach().cpu().item(),
            'allocator/concentration_max': concentration.max().detach().cpu().item(),
            'allocator/actor_loss': np.mean(actor_loss_history),
            'allocator/r2': np.maximum(np.mean(r2_history), -1),
            'allocator/log_probs': new_log_prob.detach().cpu().numpy().tolist(),
            'allocator/advantages': advantages.flatten().detach().cpu().numpy().tolist(),
        }


class NoBudgetQCritic(CustomModule):

    def __init__(self, edge_component_mapping, n_features, lr) -> None:
        super().__init__(name='NoBudgetQCritic')

        self.n_components = len(edge_component_mapping)
        self.n_edges = sum([len(v) for v in edge_component_mapping])
        self.n_features = n_features

        self.state_extractor = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.n_edges * self.n_features, 512),
            torch.nn.ReLU(),
        )

        self.action_extractor = torch.nn.Sequential(
            torch.nn.Linear(self.n_components, 512),
            torch.nn.ReLU(),
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(512 + 512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, allocation):

        return self.model(
            torch.cat((
                self.state_extractor(state),
                self.action_extractor(allocation)
            ), dim=1)
        ) + 1000.0

    def extra_repr(self) -> str:
        return f'n_components={self.n_components}, n_features={self.n_features}'


class NoBudgetDeterministicActor(CustomModule):
    def __init__(self, edge_component_mapping, n_features, lr) -> None:
        super().__init__('StochasticActor')

        self.n_components = len(edge_component_mapping)
        self.n_edges = sum([len(v) for v in edge_component_mapping])
        self.n_features = n_features

        self.state_extractor = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.n_edges * self.n_features, 512),
            torch.nn.ReLU(),
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, self.n_components),
            torch.nn.Softplus(),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        logits = self.model(
            self.state_extractor(state)
        )
        return logits

    def extra_repr(self) -> str:
        return f'n_components={self.n_components}, n_features={self.n_features}'


class TD3NoBudgetAllocator(NoBudgetAllocatorInterface):
    def __init__(self, edge_component_mapping, n_features, critic_lr, actor_lr, tau, gamma, target_allocation_noise_scale, actor_update_steps):
        super().__init__(name='TD3NoBudgetAllocator')

        self.edge_component_mapping = edge_component_mapping
        self.n_components = len(edge_component_mapping)
        self.n_features = n_features
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.tau = tau
        self.gamma = gamma
        self.target_allocation_noise_scale = target_allocation_noise_scale
        self.actor_update_steps = actor_update_steps
        self.training_step = 0

        self.actor = NoBudgetDeterministicActor(self.edge_component_mapping, self.n_features, self.actor_lr)
        self.target_actor = NoBudgetDeterministicActor(self.edge_component_mapping, self.n_features, self.actor_lr)

        self.critic = NoBudgetQCritic(self.edge_component_mapping, self.n_features, self.critic_lr)
        self.target_critic = NoBudgetQCritic(self.edge_component_mapping, self.n_features, self.critic_lr)

        self.critic_1 = NoBudgetQCritic(self.edge_component_mapping, self.n_features, self.critic_lr)
        self.target_critic_1 = NoBudgetQCritic(self.edge_component_mapping, self.n_features, self.critic_lr)

        hard_sync(self.target_actor, self.actor)
        hard_sync(self.target_critic, self.critic)
        hard_sync(self.target_critic_1, self.critic_1)

    def forward(self, states, deterministic):
        return self.actor.forward(states)

    def forward_target(self, states, deterministic):
        return self.target_actor.forward(states)

    def update(self, states, allocations_times_budgets, rewards, next_states, dones,
               truncateds):

        # Calculate Target Values
        with torch.no_grad():
            target_allocation_noise = torch.distributions.Normal(loc=0.0, scale=self.target_allocation_noise_scale)
            next_allocations_times_budgets = self.target_actor.forward(next_states)

            noisy_next_allocations = torch.maximum(next_allocations_times_budgets + target_allocation_noise.sample(next_allocations_times_budgets.shape), torch.zeros_like(next_allocations_times_budgets, device=self.device))

            next_q_values = self.target_critic.forward(next_states, noisy_next_allocations)
            next_q_values_1 = self.target_critic_1.forward(next_states, noisy_next_allocations)
            target_q_values = rewards + (1 - dones) * self.gamma * torch.minimum(next_q_values, next_q_values_1)

        # Update Critic 0
        q_values = self.critic.forward(states, allocations_times_budgets)
        critic_loss = torch.nn.functional.mse_loss(q_values, target_q_values)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        soft_sync(self.target_critic, self.critic, self.tau)

        # Update Critic 1
        q_values_1 = self.critic_1.forward(states, allocations_times_budgets)
        critic_loss_1 = torch.nn.functional.mse_loss(q_values_1, target_q_values)

        self.critic_1.optimizer.zero_grad()
        critic_loss_1.backward()
        self.critic_1.optimizer.step()
        soft_sync(self.target_critic_1, self.critic_1, self.tau)

        stats = {
            'allocator/q_loss_0': critic_loss.detach().cpu().item(),
            'allocator/q_loss_1': critic_loss_1.detach().cpu().item(),
            'allocator/q_min': q_values.min().detach().cpu().item(),
            'allocator/q_max': q_values.max().detach().cpu().item(),
            'allocator/q_mean': q_values.mean().detach().cpu().item(),
            'allocator/rewards': rewards.mean().detach().cpu().item(),
        }

        # Update actor
        self.training_step += 1
        if self.training_step % self.actor_update_steps == 0:
            current_action = self.actor.forward(states)
            current_q_values = self.critic.forward(states, current_action)

            actor_loss = - current_q_values.mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            soft_sync(self.target_actor, self.actor, self.tau)

            stats |= {
                'allocator/a_val': -actor_loss.detach().cpu().item(),
            }

        return stats


class TD3Allocator(DDPGAllocator):
    def __init__(self, edge_component_mapping, n_features, critic_lr, actor_lr, tau, gamma,
                    target_allocation_noise_scale, actor_update_steps):
        super().__init__(edge_component_mapping, n_features, critic_lr, actor_lr, tau, gamma)
        self.target_allocation_noise_scale = target_allocation_noise_scale
        self.actor_update_steps = actor_update_steps

        self.critic_1 = QCritic(self.n_components, n_features, critic_lr)
        self.target_critic_1 = QCritic(self.n_components, n_features, critic_lr)

        hard_sync(self.target_critic_1, self.critic_1)

        self.update_iteration = 0

    def update(self, aggregated_states, allocations, budgets, rewards, next_aggregated_states, next_budgets, dones,
               truncateds):

        # Calculate Target Values
        with torch.no_grad():
            target_allocation_noise = torch.distributions.Normal(loc=0.0, scale=self.target_allocation_noise_scale)
            next_allocations = self.target_actor.forward(next_aggregated_states, next_budgets)

            noisy_next_allocations = torch.nn.functional.normalize(
                torch.maximum(next_allocations + target_allocation_noise.sample(next_allocations.shape),
                              torch.zeros_like(next_allocations, device=self.device)), dim=1, p=1)

            next_q_values = self.target_critic.forward(next_aggregated_states, noisy_next_allocations, next_budgets)
            next_q_values_1 = self.target_critic_1.forward(next_aggregated_states, noisy_next_allocations, next_budgets)
            target_q_values = rewards + (1 - dones) * self.gamma * torch.minimum(next_q_values, next_q_values_1)

        # Update Critic 0
        q_values = self.critic.forward(aggregated_states, allocations, budgets)
        critic_loss = torch.nn.functional.mse_loss(q_values, target_q_values)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        soft_sync(self.target_critic, self.critic, self.tau)

        # Update Critic 1
        q_values_1 = self.critic_1.forward(aggregated_states, allocations, budgets)
        critic_loss_1 = torch.nn.functional.mse_loss(q_values_1, target_q_values)

        self.critic_1.optimizer.zero_grad()
        critic_loss_1.backward()
        self.critic_1.optimizer.step()
        soft_sync(self.target_critic_1, self.critic_1, self.tau)

        stats = {
            'allocator/q_loss': critic_loss.detach().cpu().item(),
            'allocator/q_min': q_values.min().detach().cpu().item(),
            'allocator/q_max': q_values.max().detach().cpu().item(),
            'allocator/q_mean': q_values.mean().detach().cpu().item(),
            'allocator/rewards': rewards.max().detach().cpu().item(),
            'allocator/r2': max(r2_score(target_q_values, q_values).detach().cpu().item(), -1),
        }

        # Update actor
        self.update_iteration += 1
        if self.update_iteration % self.actor_update_steps == 0:
            current_action = self.actor.forward(aggregated_states, budgets)
            current_q_values = self.critic.forward(aggregated_states, current_action, budgets)

            actor_loss = - current_q_values.mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            soft_sync(self.target_actor, self.actor, self.tau)

            stats |= {
                'allocator/a_val': -actor_loss.detach().cpu().item(),
            }

        return stats

