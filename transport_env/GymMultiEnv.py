import random
from abc import ABC
from typing import Any, SupportsFloat, List

import gymnasium as gym
import numpy as np
import torch
from gymnasium.core import ObsType, ActType

from models.agents.heuristics.attackers.allocators import ProportionalAllocator
from models.autoencoders.action_ae import Decoder, ConditionalDecoder
from transport_env.MultiAgentEnv import DynamicMultiAgentTransportationNetworkEnvironment
from util.math import sigmoid


class GymMultiEnv(gym.Env, ABC):
    def __init__(self, config):
        self.env = DynamicMultiAgentTransportationNetworkEnvironment(config)
        self.action_space = gym.spaces.Box(low=0, high=30, shape=(self.env.num_edges, ))
        self.observation_space = gym.spaces.Box(low=0, high=100_000, shape=(self.env.num_edges, 5, ))

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        o, r, d, i = self.env.step(action)
        return o, r, d, i.get('TimeLimit.truncated', False), i

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        ObsType, dict[str, Any]]:
        return self.env.reset(), {}


class ComponentTrainer(gym.Env):

    def __init__(self, config, edge_component_mapping, index):
        super().__init__()
        self.env = GymMultiEnv(config)
        self.edge_component_mapping = edge_component_mapping
        self.index = index
        self.budget = 0.0

        self.action_space = gym.spaces.Box(low=0, high=1, shape=(len(self.edge_component_mapping[self.index]), ))
        self.observation_space = gym.spaces.Box(low=0, high=100_000, shape=(len(self.edge_component_mapping[self.index]) * 5 + 1, ))

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        reconstructed_action = np.zeros(self.env.env.num_edges)
        reconstructed_action[self.edge_component_mapping[self.index]] = action / max(sum(action), 1e-8) * self.budget

        o, r, d, t, i = self.env.step(reconstructed_action)

        self.budget = np.random.uniform(0, 30)
        return np.append(o[self.edge_component_mapping[self.index]].flatten(), self.budget), r[self.index], d, t, i

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        ObsType, dict[str, Any]]:
        self.budget = np.random.uniform(0, 30)
        o, i = self.env.reset()
        return np.append(o[self.edge_component_mapping[self.index]].flatten(), self.budget), i


class ComponentTrainerWithDecoder(gym.Env):
    def __init__(self, config, edge_component_mapping, index):
        super().__init__()
        self.env = ComponentTrainer(config, edge_component_mapping, index)

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2, ))
        self.observation_space = self.env.observation_space
        self.decoder: ConditionalDecoder = torch.load(f'saved_partial_models/decoder_{index}.pt', map_location=torch.device('cpu')).decoder

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        reconstructed_action = self.decoder.forward(
            torch.tensor(self.last_obs).unsqueeze(0).float(),
            torch.tensor(action).unsqueeze(0).float()
        ).detach().cpu().numpy()[0]
        o, r, d, t, i = self.env.step(reconstructed_action)
        self.last_obs = o
        return o, r, d, t, i

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.last_obs, info = self.env.reset()
        return self.last_obs, info


class MultiEnvWithDecoder(gym.Env):

    def __init__(self, config, edge_component_mapping):
        super().__init__()

        self.env = GymMultiEnv(config)
        self.decoder: List[ConditionalDecoder] = [torch.load(f'saved_partial_models/decoder_{i}.pt', map_location=torch.device('cpu')).decoder for i in range(len(edge_component_mapping))]
        self.edge_component_mapping = edge_component_mapping

        self.proportional_allocator = ProportionalAllocator()

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2 * len(edge_component_mapping) + len(edge_component_mapping), ))
        self.observation_space = gym.spaces.Box(low=0, high=100_000, shape=(self.env.env.num_edges, 5, ))

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        reconstructed_action = np.zeros_like(self.env.action_space.sample())
        budgets = action[-len(self.edge_component_mapping):] + 1
        budgets = budgets / max(sum(budgets), 1e-8) * 30
        for i in range(len(self.edge_component_mapping)):
            obs = np.append(self.last_observation[self.edge_component_mapping[i]], budgets[i])
            reconstructed_action[self.edge_component_mapping[i]] = self.decoder[i].forward(
                torch.tensor(obs).unsqueeze(0).float(),
                torch.tensor(action[i * 2: (i + 1) * 2]).unsqueeze(0).float()
            ).detach().cpu().numpy()[0] * budgets[i]

        o, r, d, t, i = self.env.step(reconstructed_action)
        self.last_observation = o
        return o, r.sum(), d, t, i

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
         self.last_observation, info = self.env.reset()
         return self.last_observation, info


class MultiEnvWithDecoderAndDetector(MultiEnvWithDecoder):
    def __init__(self, config, edge_component_mapping):
        super().__init__(config, edge_component_mapping)

    def __get_state_values_assuming_no_action(self, done): # -> value, cumulative_reward, steps, original_reward
        gamma = 0.99

        immediate_rewards = []
        original_reward = []
        step_count = -1

        while not done:
            step_count += 1
            action = np.zeros((self.env.env.base.number_of_edges(),))
            obs, reward, done, info = self.env.env.step(action)
            original_reward.append(info.get('original_reward'))
            immediate_rewards.append(reward)

        immediate_rewards = np.array(immediate_rewards)
        state_value = np.zeros(immediate_rewards.shape[1])
        for i in range(immediate_rewards.shape[0]):
            state_value += gamma ** i * immediate_rewards[i]

        return sum(state_value), sum(np.sum(immediate_rewards, axis=0)), step_count, original_reward

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        reconstructed_action = np.zeros_like(self.env.action_space.sample())
        budgets = (action[-len(self.edge_component_mapping):] + 1) / 2
        budgets = 30 * budgets ** (np.log(0.25) / np.log(0.5))  # maps budgets to [0, 30] with 0.5 being the middle point

        if sigmoid(sum(budgets), 1.5, 30.0) > random.random():  # detected, high probability if the sum of budgets is more than 30.0, low probability if less than 30.0
            value, cumulative_reward, steps, original_reward = self.__get_state_values_assuming_no_action(False)
            return np.zeros_like(self.observation_space.sample()), value, True, False, {}
        else:
            for i in range(len(self.edge_component_mapping)):
                obs = np.append(self.last_observation[self.edge_component_mapping[i]], budgets[i])
                reconstructed_action[self.edge_component_mapping[i]] = self.decoder[i].forward(
                    torch.tensor(obs).unsqueeze(0).float(),
                    torch.tensor(action[i * 2: (i + 1) * 2]).unsqueeze(0).float()
                ).detach().cpu().numpy()[0] * budgets[i]

            o, r, d, t, i = self.env.step(reconstructed_action)
            self.last_observation = o
            return o, r.sum(), d, t, i