import logging

import numpy as np
import torch

from models import ComponentInterface, CustomModule
from util.torch.math import r2_score
from util.torch.misc import hard_sync, soft_sync
from util.torch.rl import GeneralizedAdvantageEstimation


class QCritic(CustomModule):

    def __init__(self, name, n_edges, n_features, lr) -> None:
        super().__init__(name)
        self.lr = lr
        self.n_edges = n_edges
        self.n_features = n_features

        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_edges * n_features + n_edges + 2, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, observation, budget, allocation, action):
        return self.model(
            torch.cat(
                (torch.flatten(observation, start_dim=1), budget, allocation, action), dim=1
            )
        )

    def extra_repr(self) -> str:
        return f'n_edges={self.n_edges}, n_features={self.n_features}, lr={self.lr}'


class VCritic(CustomModule):
    def __init__(self, name, n_edges, n_features) -> None:
        super().__init__(name)
        self.n_edges = n_edges
        self.n_features = n_features

        self.state_extractor = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(n_edges * n_features, 64),
            torch.nn.ReLU(),
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(64 + 2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, observation, budget, allocation):
        return self.model(
            torch.cat(
                (self.state_extractor(observation), budget, allocation), dim=1
            )
        )

    def extra_repr(self) -> str:
        return f'n_edges={self.n_edges}, n_features={self.n_features}'


class DeterministicActor(CustomModule):

    def __init__(self, name, n_edges, n_features, lr) -> None:
        super().__init__(name)
        self.lr = lr
        self.n_edges = n_edges
        self.n_features = n_features

        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_edges * n_features + 2, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, n_edges),
            # torch.nn.Softmax(dim=1),
            # torch.nn.ReLU()
            torch.nn.Sigmoid()
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, observation, budget, allocation, deterministic):
        logits = self.model(
            torch.cat(
                (torch.flatten(observation, start_dim=1), allocation, budget), dim=1
            )
        )

        return torch.nn.functional.normalize(logits, dim=1, p=1)

    def extra_repr(self) -> str:
        return f'n_edges={self.n_edges}, n_features={self.n_features}, lr={self.lr}'


class StochasticActor(CustomModule):

    def __init__(self, name, n_edges, n_features, max_concentration) -> None:
        super().__init__(name)
        self.max_concentration = max_concentration
        self.n_edges = n_edges
        self.n_features = n_features

        self.state_extractor = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(n_edges * n_features, 128),
            torch.nn.ReLU(),
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(128 + 2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_edges),
            # torch.nn.Softmax(dim=1),
            # torch.nn.ReLU()
            # torch.nn.Sigmoid()
            torch.nn.ReLU()
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
        return f'n_edges={self.n_edges}, n_features={self.n_features}, max_concentration={self.max_concentration}'


class DDPGComponent(ComponentInterface):
    def __init__(self, edge_component_mapping, n_features, critic_lr, actor_lr, tau, gamma, noise) -> None:
        super().__init__(name='DDPGComponent')
        self.tau = tau
        self.gamma = gamma
        self.edge_component_mapping = edge_component_mapping
        self.n_components = len(edge_component_mapping)
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.noise = noise

        self.critics = torch.nn.ModuleList([
            QCritic(f'Critic-{i}', len(edge_component_mapping[i]), n_features, critic_lr) for i
            in
            range(self.n_components)
        ])
        self.actors = torch.nn.ModuleList([
            DeterministicActor(f'Actor-{i}', len(edge_component_mapping[i]), n_features, actor_lr) for i
            in
            range(self.n_components)
        ])

        self.target_critics = torch.nn.ModuleList([
            QCritic(f'Critic-{i}', len(edge_component_mapping[i]), n_features, critic_lr) for i
            in
            range(self.n_components)
        ])
        self.target_actors = torch.nn.ModuleList([
            DeterministicActor(f'Actor-{i}', len(edge_component_mapping[i]), n_features, actor_lr) for i
            in
            range(self.n_components)
        ])

        hard_sync(self.target_critics, self.critics)
        hard_sync(self.target_actors, self.actors)

        self.factorized_critic_optimizer = torch.optim.Adam(self.parameters(), lr=critic_lr)

    def __forward_critic(self, critics, states, budgets, allocations, actions):
        sum_of_critics = torch.empty((states.shape[0], 1), device=self.device)
        for c in range(self.n_components):
            sum_of_critics += critics[c].forward(
                states[:, self.edge_component_mapping[c], :],
                budgets,
                allocations[:, [c]],
                actions[:, self.edge_component_mapping[c]]
            )
        return sum_of_critics

    def forward_critic(self, states, budgets, allocations, actions):
        return self.__forward_critic(self.critics, states, budgets, allocations, actions)

    def forward_target_critic(self, states, budgets, allocations, actions):
        return self.__forward_critic(self.target_critics, states, budgets, allocations, actions)

    def __forward_actor(self, actors, states, budgets, allocations, deterministic):
        actions = torch.empty(
            (states.shape[0], sum([len(c) for c in self.edge_component_mapping])),
            device=self.device)

        for c in range(self.n_components):
            component_action = actors[c].forward(
                states[:, self.edge_component_mapping[c], :],
                budgets,
                allocations[:, [c]],
                deterministic=deterministic
            )
            actions[:, self.edge_component_mapping[c]] = component_action

        return actions

    def forward_actor(self, states, budgets, allocations, deterministic):
        return self.__forward_actor(self.actors, states, budgets, allocations, deterministic)

    def forward_target_actor(self, states, budgets, allocations, deterministic):
        return self.__forward_actor(self.target_actors, states, budgets, allocations, deterministic)

    def forward(self, states, budgets, allocations, deterministic):
        action = self.forward_actor(states, budgets, allocations, deterministic)
        if not deterministic:
            return self.__normalize_action(action + self.noise(action.shape), allocations, budgets)
        else:
            return action

    def __normalize_action(self, action, allocations, budget):
        action = torch.maximum(action, torch.zeros_like(action))
        for c in range(self.n_components):
            action[:, self.edge_component_mapping[c]] = torch.nn.functional.normalize(action[:, self.edge_component_mapping[c]], p=1, dim=1) * allocations[:, [c]] * budget
        return action

    def __update_critics(self, index, states, actions, budgets, allocations, next_states, next_budgets, next_allocations, rewards, dones):

        # updating critic

        with torch.no_grad():
            target_action = self.target_actors[index].forward(
                next_states[:, self.edge_component_mapping[index], :],
                next_budgets,
                next_allocations[:, [index]],
                deterministic=False
            )
            component_target_q_values = self.target_critics[index].forward(
                next_states[:, self.edge_component_mapping[index], :],
                next_budgets,
                next_allocations[:, [index]],
                target_action
            )

            component_y = rewards[:, [index]] + self.gamma * component_target_q_values * (1 - dones)

        component_current_q = self.critics[index].forward(
            states[:, self.edge_component_mapping[index], :],
            budgets,
            allocations[:, [index]],
            actions[:, self.edge_component_mapping[index]]
        )

        component_critic_loss = torch.nn.functional.mse_loss(
            component_y,
            component_current_q
        )
        self.critics[index].optimizer.zero_grad()
        component_critic_loss.backward()
        self.critics[index].optimizer.step()

        stat = dict(
            q_loss=component_critic_loss.cpu().data.numpy().item(),
            q_r2=max(r2_score(component_y, component_current_q).cpu().data.numpy().item(), -1),
            q_max=component_y.max().cpu().data.numpy().item(),
            q_min=component_y.min().cpu().data.numpy().item(),
            q_mean=component_y.mean().cpu().data.numpy().item(),
        )

        return stat

    def __update_actors(self, index, states, allocations, budgets):
        # updating actor

        component_current_action = self.actors[index].forward(
            states[:, self.edge_component_mapping[index], :],
            budgets,
            allocations[:, [index]],
            deterministic=False
        )

        current_value = self.critics[index].forward(
                states[:, self.edge_component_mapping[index], :],
                budgets,
                allocations[:, [index]],
                component_current_action
            )

        component_actor_loss = -torch.mean(
            current_value
        )

        self.actors[index].optimizer.zero_grad()
        component_actor_loss.backward()
        self.actors[index].optimizer.step()

        return dict(
            a_val=-component_actor_loss.cpu().data.numpy().item(),
            noise=self.noise.get_current_noise().cpu().data.numpy().item()
        )

    def __update_multi_agent(self, states, actions, budgets, allocations, next_states, next_budgets, next_allocations, rewards, dones):
        stats = dict(
            q_loss=np.zeros(self.n_components),
            q_r2=np.zeros(self.n_components),
            q_max=np.zeros(self.n_components),
            q_min=np.zeros(self.n_components),
            q_mean=np.zeros(self.n_components),
            a_val=np.zeros(self.n_components),
            noise=np.zeros(self.n_components),
        )

        for c in range(self.n_components):
            # Updating component critic

            stat = self.__update_critics(
                c,
                states,
                actions,
                budgets,
                allocations,
                next_states,
                next_budgets,
                next_allocations,
                rewards,
                dones
            )

            for k, v in stat.items():
                stats[k][c] = v

            stat = self.__update_actors(
                c,
                states,
                allocations,
                budgets
            )

            for k, v in stat.items():
                stats[k][c] = v

        soft_sync(self.target_critics, self.critics, self.tau)
        soft_sync(self.target_actors, self.actors, self.tau)

        returnable_stats = dict()
        for k, v in stats.items():
            for c in range(self.n_components):
                returnable_stats[f'component/{k}/{c}'] = v[c]
        return returnable_stats | {
            'component/noise': self.noise.get_current_noise(),
        }

    def update(self, states, actions, budgets, allocations, next_states, next_budgets, next_allocations, rewards, dones, truncateds):
        return self.__update_multi_agent(
            states, actions, budgets, allocations, next_states, next_budgets, next_allocations, rewards, dones)


class PPOComponentAgent(CustomModule):
    def __init__(self, index, n_edges, n_features, actor_lr, critic_lr, gamma, lam, epsilon, policy_grad_clip, value_coeff, entropy_coeff, n_updates, batch_size, max_concentration, clip_range_vf) -> None:
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
        self.max_concentration = max_concentration
        self.clip_range_vf = clip_range_vf

        self.value = VCritic(f'Value-{index}', n_edges, n_features)
        self.actor = StochasticActor(f'Actor-{index}', n_edges, n_features, max_concentration)

        self.gae = GeneralizedAdvantageEstimation(self.gamma, self.lam)
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': self.actor_lr},
            {'params': self.value.parameters(), 'lr': self.critic_lr},
        ])

    def extra_repr(self) -> str:
        return f'n_features={self.n_features}, actor_lr={self.actor_lr}, critic_lr={self.critic_lr}, gamma={self.gamma}, lam={self.lam}, epsilon={self.epsilon}, policy_grad_clip={self.policy_grad_clip}, value_coeff={self.value_coeff}, entropy_coeff={self.entropy_coeff}, n_updates={self.n_updates}, batch_size={self.batch_size}, max_concentration={self.max_concentration}, clip_range_vf={self.clip_range_vf}'

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
                batch_advantages = (advantages[indices] - advantages[indices].mean()) / (advantages[indices].std())
                distribution = self.actor.get_distribution(states[indices], budgets[indices],
                                                                           allocations[indices])
                new_log_prob = torch.unsqueeze(distribution.log_prob(actions[indices]), dim=1)
                # ratio = torch.exp(new_log_prob - old_log_prob[indices])
                ratio = new_log_prob - old_log_prob[indices]
                surr1 = ratio * batch_advantages
                # surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                surr2 = torch.clamp(ratio, -self.epsilon, self.epsilon) * batch_advantages
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
            f'component/{self.index}/kl_divergence_mean': kl_divergence.mean().detach().cpu().item(),
            f'component/{self.index}/kl_divergence_max': kl_divergence.max().detach().cpu().item(),
            f'component/{self.index}/entropy_mean': entropy.mean().detach().cpu().item(),
            f'component/{self.index}/concentration_mean': concentration.mean().detach().cpu().item(),
            f'component/{self.index}/concentration_min': concentration.min().detach().cpu().item(),
            f'component/{self.index}/concentration_max': concentration.max().detach().cpu().item(),
            f'component/{self.index}/log_ratio_mean': ratio.mean().detach().cpu().item(),
            f'component/{self.index}/log_ratio_max': ratio.max().detach().cpu().item(),
            f'component/{self.index}/log_ratio_min': ratio.min().detach().cpu().item(),
            f'component/{self.index}/actor_loss': np.mean(actor_loss_history),
            f'component/{self.index}/r2': np.maximum(np.mean(r2_history), -1),
            f'component/{self.index}/log_probs': new_log_prob.detach().cpu().numpy().tolist(),
            f'component/{self.index}/advantages': advantages.flatten().detach().cpu().numpy().tolist(),
        }


class PPOComponent(ComponentInterface):

    def __init__(self, edge_component_mapping, n_features, actor_lr, critic_lr, gamma, lam, epsilon, policy_grad_clip, value_coeff, entropy_coeff, n_updates, batch_size, max_concentration, clip_range_vf) -> None:
        super().__init__('PPOComponent')
        self.edge_component_mapping = edge_component_mapping
        self.n_components = len(edge_component_mapping)

        self.agents = torch.nn.ModuleList([
            PPOComponentAgent(
                i, len(edge_component_mapping[i]), n_features, actor_lr, critic_lr, gamma, lam, epsilon, policy_grad_clip,
                value_coeff, entropy_coeff, n_updates, batch_size, max_concentration, clip_range_vf
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

    def update(self, states, actions, budgets, allocations, next_states, next_budgets, next_allocations, rewards, dones,
               truncateds):
        stats = {}
        for c in range(self.n_components):
            stats |= self.agents[c].update(
                states[:, self.edge_component_mapping[c], :],
                actions[:, self.edge_component_mapping[c]],
                budgets,
                allocations[:, [c]],
                rewards[:, [c]],
                dones,
                truncateds
            )

        return stats


class TD3Component(DDPGComponent):

    def __init__(self, edge_component_mapping, n_features, critic_lr, actor_lr, tau, gamma, noise, target_action_noise_scale, actor_update_steps) -> None:
        super().__init__(edge_component_mapping, n_features, critic_lr, actor_lr, tau, gamma, noise)

        self.target_action_noise_scale = target_action_noise_scale
        self.actor_update_steps = actor_update_steps

        self.critics_1 = torch.nn.ModuleList([
            QCritic(f'Critic_1-{i}', len(edge_component_mapping[i]), n_features, critic_lr) for i
            in
            range(self.n_components)
        ])

        self.target_critics_1 = torch.nn.ModuleList([
            QCritic(f'Critic_1-{i}', len(edge_component_mapping[i]), n_features, critic_lr) for i
            in
            range(self.n_components)
        ])

        hard_sync(self.target_critics_1, self.critics_1)

        self.training_step = 0

    def forward_critic_1(self, states, budgets, allocations, actions):
        return self.__forward_critic(self.critics_1, states, budgets, allocations, actions)

    def __update_critics(self, index, states, actions, budgets, allocations, next_states, next_budgets, next_allocations, rewards, dones):

        # updating critic

        with torch.no_grad():

            target_action_noise_distribution = torch.distributions.Normal(loc=0.0, scale=self.target_action_noise_scale)

            target_action = self.target_actors[index].forward(
                next_states[:, self.edge_component_mapping[index], :],
                next_budgets,
                next_allocations[:, [index]],
                deterministic=False
            )

            target_action = torch.nn.functional.normalize(
                torch.maximum(target_action + target_action_noise_distribution.sample(target_action.shape), torch.zeros_like(target_action, device=self.device)), p=1, dim=1
            )

            component_target_q_values = self.target_critics[index].forward(
                next_states[:, self.edge_component_mapping[index], :],
                next_budgets,
                next_allocations[:, [index]],
                target_action
            )
            component_target_q_values_1 = self.target_critics_1[index].forward(
                next_states[:, self.edge_component_mapping[index], :],
                next_budgets,
                next_allocations[:, [index]],
                target_action
            )

            component_y = rewards[:, [index]] + self.gamma * torch.minimum(component_target_q_values, component_target_q_values_1) * (1 - dones)

        # Update Critic 0

        component_current_q = self.critics[index].forward(
            states[:, self.edge_component_mapping[index], :],
            budgets,
            allocations[:, [index]],
            actions[:, self.edge_component_mapping[index]]
        )

        component_critic_loss = torch.nn.functional.mse_loss(
            component_y,
            component_current_q
        )
        self.critics[index].optimizer.zero_grad()
        component_critic_loss.backward()
        self.critics[index].optimizer.step()

        # Update Critic 1

        component_current_q_1 = self.critics_1[index].forward(
            states[:, self.edge_component_mapping[index], :],
            budgets,
            allocations[:, [index]],
            actions[:, self.edge_component_mapping[index]]
        )

        component_critic_loss_1 = torch.nn.functional.mse_loss(
            component_y,
            component_current_q_1
        )

        self.critics_1[index].optimizer.zero_grad()
        component_critic_loss_1.backward()
        self.critics_1[index].optimizer.step()

        stat = dict(
            q_loss=component_critic_loss.cpu().data.numpy().item(),
            q_r2=max(r2_score(component_y, component_current_q).cpu().data.numpy().item(), -1),
            q_max=component_y.max().cpu().data.numpy().item(),
            q_min=component_y.min().cpu().data.numpy().item(),
            q_mean=component_y.mean().cpu().data.numpy().item(),
        )

        return stat

    def __update_actors(self, index, states, allocations, budgets):
        self.training_step += 1
        if self.training_step % self.actor_update_steps == 0:
            return super().__update_actors(index, states, allocations, budgets)
        return dict()
