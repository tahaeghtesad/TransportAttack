import logging

import numpy as np
import torch

from util.torch.math import r2_score


class LowLevelCritic(torch.nn.Module):

    def __init__(self, name, config, observation_space_shape, action_space_shape) -> None:
        super().__init__()
        self.logger = logging.getLogger(f'{name}')
        self.config = config

        self.observation_in = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(np.prod(observation_space_shape), 64),
            torch.nn.ReLU(),
        )

        self.action_in = torch.nn.Sequential(
            torch.nn.Linear(np.prod(action_space_shape), 64),
            torch.nn.ReLU(),
        )

        self.combined = torch.nn.Sequential(
            torch.nn.Linear(64 + 64 + 1, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            # torch.nn.ReLU()
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config['rl_config']['low_level_lr']['critic'])

    def forward(self, observation, budget, action):
        return self.combined(
            torch.cat(
                (self.observation_in(observation), budget, self.action_in(action)), dim=1
            )
        )


class LowLevelActor(torch.nn.Module):
    # input: observation (o)
    # output: A vector of action space shape (a)
    # update rule: should be to maximize the critic value with gradient ascent

    def __init__(self, name, config, observation_space_shape, action_space_shape) -> None:
        super().__init__()
        self.logger = logging.getLogger(f'{name}')
        self.config = config

        self.state_in = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(np.prod(observation_space_shape), 128),
            torch.nn.ReLU(),
        )  # TODO make it different blocks for each edge feature

        self.output = torch.nn.Sequential(
            torch.nn.Linear(128 + 2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, np.prod(action_space_shape)),
            torch.nn.Softplus()
            # torch.nn.ReLU()
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config['rl_config']['low_level_lr']['actor'])

    def forward(self, observation, budget, magnitude):
        logits = self.output(
            torch.cat(
                (self.state_in(observation), budget, magnitude), dim=1
            )
        )

        # TODO if the output is more than the component budge * magnitude, rescale it.
        return torch.nn.functional.normalize(logits, p=1, dim=1) * budget * magnitude


class HighLevelCritic(torch.nn.Module):

    def __init__(self, config, env) -> None:
        super().__init__()
        self.logger = logging.getLogger('HighLevelCritic')
        self.config = config

        self.n_components = env.n_components

        self.state_model = torch.nn.Sequential(
            torch.nn.Linear(self.n_components * 5, 128),
            torch.nn.ReLU(),
        )

        # self.vehicle_model = torch.nn.Sequential(
        #     torch.nn.Linear(self.n_components * 2, 128),
        #     torch.nn.ReLU(),
        # )

        self.output = torch.nn.Sequential(
            torch.nn.Linear(128 + self.n_components + 1, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config['rl_config']['high_level_lr']['critic'])

    def forward(self, aggregated_state, vehicles_in_component, component_budget, magnitude):

        return self.output(
            torch.cat((
                self.state_model(torch.flatten(aggregated_state, start_dim=1)),
                # self.vehicle_model(torch.flatten(vehicles_in_component, start_dim=1)),
                component_budget,
                magnitude
            ), dim=1)
        )


class HighLevelActor(torch.nn.Module):

    def __init__(self, env, config) -> None:
        super().__init__()
        self.logger = logging.getLogger('HighLevelActor')
        self.config = config

        self.n_components = env.n_components
        self.edge_component_mapping = env.edge_component_mapping

        self.state_model = torch.nn.Sequential(
            torch.nn.Linear(self.n_components * 5, 128),
            torch.nn.ReLU(),
        )

        # self.vehicle_model = torch.nn.Sequential(
        #     torch.nn.Linear(self.n_components * 2, 128),
        #     torch.nn.ReLU(),
        # )

        self.budget_model = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.n_components),
            torch.nn.Sigmoid()
        )

        self.magnitude_model = torch.nn.Sequential(
            torch.nn.Linear(128, 1),
            torch.nn.Softplus(),
            # torch.nn.Sigmoid()
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config['rl_config']['high_level_lr']['actor'])

    def forward(self, aggregated_state, vehicles_in_component):

        # features = torch.cat((
        #         self.state_model(torch.flatten(aggregated_state, start_dim=1)),
        #         self.vehicle_model(torch.flatten(vehicles_in_component, start_dim=1)),
        #     ), dim=1)

        features = self.state_model(torch.flatten(aggregated_state, start_dim=1))

        return (
            torch.nn.functional.normalize(self.budget_model(features), dim=1, p=1),
            self.magnitude_model(features)
        )


class MADDPGModel(torch.nn.Module):
    def __init__(self, config, env, device) -> None:
        super().__init__()
        self.logger = logging.getLogger('MADDPGModel')
        self.n_components = env.n_components
        self.edge_component_mapping = env.edge_component_mapping
        self.config = config

        self.critics = torch.nn.ModuleList([
            LowLevelCritic(f'Critic-{i}', self.config, env.observation_space[i].shape, env.action_space[i].shape) for i
            in
            range(self.n_components)
        ])
        self.actors = torch.nn.ModuleList([
            LowLevelActor(f'Actor-{i}', self.config, env.observation_space[i].shape, env.action_space[i].shape) for i in
            range(self.n_components)
        ])

        self.target_critics = torch.nn.ModuleList([
            LowLevelCritic(f'TargetCritic-{i}', self.config, env.observation_space[i].shape, env.action_space[i].shape)
            for i in
            range(self.n_components)
        ])
        self.target_actors = torch.nn.ModuleList([
            LowLevelActor(f'TargetActor-{i}', self.config, env.observation_space[i].shape, env.action_space[i].shape)
            for i in
            range(self.n_components)
        ])

        self.high_level_critic = HighLevelCritic(self.config, env)
        self.high_level_actor = HighLevelActor(env, self.config)

        self.high_level_target_critic = HighLevelCritic(self.config, env)
        self.high_level_target_actor = HighLevelActor(env, self.config)

        self.criterion = torch.nn.MSELoss()

        self.logger.info(f'Total parameters: {sum(p.numel() for p in self.parameters())}')

        self.tau = self.config['rl_config']['tau']
        self.gamma = self.config['rl_config']['gamma']
        self.device = device
        self.to(self.device)
        self.hard_sync()

    def hard_sync(self):
        for critic, target_critic in zip(self.critics, self.target_critics):
            for param, target_param in zip(critic.parameters(), target_critic.parameters()):
                target_param.data.copy_(param.data)

        for actor, target_actor in zip(self.actors, self.target_actors):
            for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                target_param.data.copy_(param.data)

        for param, target_param in zip(self.high_level_critic.parameters(), self.high_level_target_critic.parameters()):
            target_param.data.copy_(param.data)

        for param, target_param in zip(self.high_level_actor.parameters(), self.high_level_target_actor.parameters()):
            target_param.data.copy_(param.data)

    def soft_sync(self):
        for critic, target_critic in zip(self.critics, self.target_critics):
            for param, target_param in zip(critic.parameters(), target_critic.parameters()):
                target_param.data.copy_(
                    (1 - self.tau) * target_param.data + self.tau * param.data
                )

        for actor, target_actor in zip(self.actors, self.target_actors):
            for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                target_param.data.copy_(
                    (1 - self.tau) * target_param.data + self.tau * param.data
                )

        for param, target_param in zip(self.high_level_critic.parameters(), self.high_level_target_critic.parameters()):
            target_param.data.copy_(
                (1 - self.tau) * target_param.data + self.tau * param.data
            )

        for param, target_param in zip(self.high_level_actor.parameters(), self.high_level_target_actor.parameters()):
            target_param.data.copy_(
                (1 - self.tau) * target_param.data + self.tau * param.data
            )

    def __get_vehicles_in_component(self, states):
        vehicles_in_component = torch.empty((states.shape[0], self.n_components, 2), device=self.device)
        for c in range(self.n_components):
            vehicles_in_component[:, [c], 0] = torch.sum(
                # number of vehicles currently on 'e'
                states[:, self.edge_component_mapping[c], 2], dim=1, keepdim=True)
            vehicles_in_component[:, [c], 1] = torch.sum(
                # number of vehicles taking 'e' imm
                states[:, self.edge_component_mapping[c], 1], dim=1, keepdim=True)

        return vehicles_in_component

    def __aggregate_state(self, states):
        aggregated = torch.empty((states.shape[0], self.n_components, 5), device=states.device)
        for c in range(self.n_components):
            aggregated[:, c, :] = torch.sum(
                states[:, self.edge_component_mapping[c]], dim=1
            )
        return aggregated

    def __get_action_budget_and_magnitude(self, actions):
        component_budget = torch.empty((actions.shape[0], self.n_components), device=self.device)
        for c in range(self.n_components):
            component_budget[:, c] = torch.sum(actions[:, self.edge_component_mapping[c]], dim=1)
        return torch.nn.functional.normalize(component_budget, p=1, dim=1), torch.norm(component_budget, p=1, dim=1, keepdim=True)

    def __update_low_level_critic(self, index, states, actions, next_states, rewards, dones):

        # mask_attack selects samples before the attack is detected
        # mask_attack = torch.squeeze(torch.eq(rewards[:, [self.n_components + 1]], 0))
        # mask done is when the episode ended. it always ends with a 0 reward
        # mask_done = torch.squeeze(torch.eq(dones, 0))
        # mask = torch.logical_or(mask_attack, mask_done)

        # updating critic

        with torch.no_grad():
            component_budget, component_budget_magnitude = self.__get_action_budget_and_magnitude(actions)
            component_target_budget, component_target_budget_magnitude = self.high_level_target_actor.forward(
                self.__aggregate_state(next_states),
                self.__get_vehicles_in_component(next_states)
            )
            component_target_action = self.target_actors[index].forward(
                next_states[:, self.edge_component_mapping[index], :],
                component_target_budget[:, [index]],
                component_target_budget_magnitude
            )
            component_target_q_values = self.target_critics[index].forward(
                next_states[:, self.edge_component_mapping[index], :],
                component_target_budget[:, [index]],
                component_target_action
            )

            component_y = rewards[:, [index]] + self.gamma * component_target_q_values * (1 - dones)

        component_current_q = self.critics[index].forward(
            states[:, self.edge_component_mapping[index], :],
            component_budget[:, [index]],
            actions[:, self.edge_component_mapping[index]]
        )

        component_critic_loss = self.criterion(
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

    def __update_low_level_actor(self, index, states, actions):
        # updating actor

        with torch.no_grad():
            component_budget, component_budget_magnitude = self.__get_action_budget_and_magnitude(actions)
            component_current_action = self.actors[index].forward(
                states[:, self.edge_component_mapping[index], :],
                component_budget[:, [index]],
                component_budget_magnitude
            )

        current_value = self.critics[index].forward(
                states[:, self.edge_component_mapping[index], :],
                component_budget[:, [index]],
                component_current_action
            )

        component_actor_loss = -torch.mean(
            current_value
        )

        self.actors[index].optimizer.zero_grad()
        component_actor_loss.backward()
        self.actors[index].optimizer.step()

        return dict(
            a_loss=component_actor_loss.cpu().data.numpy().item(),
        )

    '''
    TODO we can calculate the value of the states after the attack is detected and use that as the target 
    '''
    def __update_high_level_critic(self, states, actions, next_states, rewards, dones):
        # Updating high level critic

        with torch.no_grad():

            next_aggregated_state = self.__aggregate_state(next_states)
            aggregated_state = self.__aggregate_state(states)

            next_vehicle_in_component = self.__get_vehicles_in_component(next_states)
            vehicle_in_component = self.__get_vehicles_in_component(states)

            target_budget, target_budget_magnitude = self.high_level_target_actor.forward(next_aggregated_state, next_vehicle_in_component)
            budget, magnitude = self.__get_action_budget_and_magnitude(actions)

            high_level_target_q_values = self.high_level_target_critic.forward(
                next_aggregated_state,
                next_vehicle_in_component,
                target_budget,
                target_budget_magnitude
            )

            # rewards[:, [self.n_components]] is whether the attack is detected or not
            high_level_y = torch.sum(
                rewards[:, :self.n_components], dim=1,
                keepdim=True
            ) + self.gamma * high_level_target_q_values * (1 - dones) * (1 - rewards[:, [self.n_components]])

        high_level_current_q = self.high_level_critic.forward(
            aggregated_state,
            vehicle_in_component,
            budget,
            magnitude
        )
        high_level_critic_loss = self.criterion(high_level_y, high_level_current_q)

        self.high_level_critic.optimizer.zero_grad()
        high_level_critic_loss.backward()
        self.high_level_critic.optimizer.step()

        stat = dict(
            high_level_loss=high_level_critic_loss.cpu().data.numpy().item(),
            high_level_r2=max(r2_score(high_level_y, high_level_current_q).cpu().data.numpy().item(), -1),
            high_level_max_q=high_level_y.max().cpu().data.numpy().item(),
            high_level_min_q=high_level_y.min().cpu().data.numpy().item(),
            high_level_mean_q=high_level_y.mean().cpu().data.numpy().item(),
        )

        return stat

    def __update_high_level_actor(self, states):
        # Updating high level actor

        with torch.no_grad():

            aggregated_state = self.__aggregate_state(states)
            vehicle_in_component = self.__get_vehicles_in_component(states)

        current_budget, current_budget_magnitude = self.high_level_actor.forward(aggregated_state, vehicle_in_component)
        current_value = self.high_level_critic.forward(
            aggregated_state,
            vehicle_in_component,
            current_budget,
            current_budget_magnitude
        )
        high_level_actor_loss = -torch.mean(current_value)

        self.high_level_actor.optimizer.zero_grad()
        high_level_actor_loss.backward()
        self.high_level_actor.optimizer.step()

        return dict(
            high_level_a_loss=high_level_actor_loss.cpu().data.numpy().item(),
        )

    def update_multi_agent(self, states, actions, next_states, rewards, dones):

        # moving data to computational device
        states = torch.from_numpy(np.array(states)).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).float().to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        dones = torch.unsqueeze(torch.from_numpy(np.array(dones)).float(), dim=1).to(self.device)

        # Updating component critics

        stats = dict(
            q_loss=[[] for _ in range(self.n_components)],
            q_r2=[[] for _ in range(self.n_components)],
            q_max=[[] for _ in range(self.n_components)],
            q_min=[[] for _ in range(self.n_components)],
            q_mean=[[] for _ in range(self.n_components)],
            a_loss=[[] for _ in range(self.n_components)],
            high_level_loss=[],
            high_level_r2=[],
            high_level_max_q=[],
            high_level_min_q=[],
            high_level_mean_q=[],
            high_level_a_loss=[]
        )

        for c in range(self.n_components):
            # Updating component critic

            stat = self.__update_low_level_critic(
                c,
                states,
                actions,
                next_states,
                rewards,
                dones
            )

            for k, v in stat.items():
                stats[k][c].append(v)

            stat = self.__update_low_level_actor(
                c,
                states,
                actions
            )

            for k, v in stat.items():
                stats[k][c].append(v)

        # Updating high level critic

        stat = self.__update_high_level_critic(
            states,
            actions,
            next_states,
            rewards,
            dones
        )

        for k, v in stat.items():
            stats[k].append(v)

        stat = self.__update_high_level_actor(states)

        for k, v in stat.items():
            stats[k].append(v)

        self.soft_sync()

        return stats

    # def update_factorized(self, states_dict, actions, next_states_dict, rewards, dones):
    #
    #     states = torch.from_numpy(np.array([s['feature_vector'] for s in states_dict])).float().to(self.device)
    #     allocations = torch.from_numpy(np.array([s['allocation'] for s in states_dict])).float().to(self.device)
    #     actions = torch.from_numpy(np.array(actions)).float().to(self.device)
    #
    #     next_states = torch.from_numpy(np.array([s['feature_vector'] for s in next_states_dict])).float().to(self.device)
    #     next_allocations = torch.from_numpy(np.array([s['allocation'] for s in next_states_dict])).float().to(self.device)
    #
    #     rewards = torch.sum(torch.from_numpy(np.array(rewards)).float().to(self.device), dim=1, keepdim=True)
    #     dones = torch.unsqueeze(torch.from_numpy(np.array(dones)).float(), dim=1).to(self.device)
    #
    #     # Update critic
    #     target_actions = self.forward_target_actor(next_states, next_allocations)
    #     target_q_values = self.forward_target_critic(next_states, next_allocations, target_actions)
    #     y = rewards + self.gamma * target_q_values * (1 - dones)
    #     current_q = self.forward_critic(states, allocations, actions)
    #     critic_loss = self.criterion(y, current_q)
    #
    #     self.critic_optimizer.zero_grad()
    #     critic_loss.backward()
    #     self.critic_optimizer.step()
    #
    #     # Update actor
    #     current_actions = self.forward_actor(states, allocations)
    #     actor_loss = -torch.mean(self.forward_critic(states, allocations, current_actions))
    #
    #     self.actor_optimizer.zero_grad()
    #     actor_loss.backward()
    #     self.actor_optimizer.step()
    #
    #
    #     self.soft_sync()
    #
    #     return dict(
    #         loss=critic_loss.cpu().data.numpy(),
    #         actor_loss=actor_loss.cpu().data.numpy(),
    #         q_values=target_q_values.cpu().data.numpy(),
    #         r2=r2_score(target_q_values, current_q).cpu().data.numpy(),
    #         max_q=target_q_values.max().cpu().data.numpy(),
    #         min_q=target_q_values.min().cpu().data.numpy(),
    #         mean_q=target_q_values.mean().cpu().data.numpy(),
    #     )

    def __forward_critic(self, critics, states, allocations, actions):
        sum_of_critics = torch.empty((states.shape[0], 1), device=self.device)
        for c in range(self.n_components):
            sum_of_critics += critics[c].forward(
                states[:, self.edge_component_mapping[c], :],
                allocations[:, [c]],
                actions[:, self.edge_component_mapping[c]]
            )
        return sum_of_critics

    def forward_critic(self, states, allocations, actions):
        return self.__forward_critic(self.critics, states, allocations, actions)

    def forward_target_critic(self, states, allocations, actions):
        return self.__forward_critic(self.target_critics, states, allocations, actions)

    def __forward_actor(self, actors, states, allocations, magnitudes):
        actions = torch.empty(
            (states.shape[0], sum([len(c) for c in self.edge_component_mapping])),
            device=self.device)

        for c in range(self.n_components):
            actions[:, self.edge_component_mapping[c]] = actors[c].forward(
                states[:, self.edge_component_mapping[c], :],
                allocations[:, [c]],
                magnitudes
            )

        return actions

    def forward_actor(self, states, allocations, magnitudes):
        return self.__forward_actor(self.actors, states, allocations, magnitudes)

    def forward_target_actor(self, states, allocations, magnitudes):
        return self.__forward_actor(self.target_actors, states, allocations, magnitudes)

    def forward(self, state):
        with torch.no_grad():
            state = torch.unsqueeze(torch.from_numpy(state).float(), dim=0).to(self.device)

            budgets, magnitudes = self.high_level_target_actor.forward(
                self.__aggregate_state(state),
                self.__get_vehicles_in_component(state)
            )

            current_actions = self.forward_actor(state, budgets, magnitudes)

        return (
            current_actions.cpu().data.numpy()[0],
            budgets.cpu().data.numpy()[0],
            magnitudes.cpu().data.numpy()[0][0]
        )

    def save(self, path):
        torch.save(self.state_dict(), path)