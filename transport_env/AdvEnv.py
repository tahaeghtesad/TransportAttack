import logging
from abc import ABC
from collections import deque
from typing import SupportsFloat, Any

import gymnasium as gym
import numpy as np
from gym.core import ActType, ObsType

from strategies.attacker_strategies import MixedAttackerStrategy
from strategies.defender_strategies import MixedDefenderStrategy
from transport_env.MultiAgentEnv import DynamicMultiAgentTransportationNetworkEnvironment


# Defender

class BaseAdvEnv(gym.Env, ABC):

    def __init__(self, config):
        super().__init__()
        self.env = DynamicMultiAgentTransportationNetworkEnvironment(config)
        self.logger = logging.getLogger(__name__)
        self.step_count = 0

    def get_state_values_assuming_no_action(self, done):
        gamma = 0.99

        truncated = False
        immediate_rewards = []
        original_reward = []
        step_count = -1

        while not done and not truncated:
            step_count += 1
            action = np.zeros((self.env.base.number_of_edges(),))
            obs, reward, done, info = self.env.step(action)
            truncated = info.get('TimeLimit.truncated', False)
            original_reward.append(info.get('original_reward'))
            immediate_rewards.append(sum(reward))

        immediate_rewards = np.array(immediate_rewards)
        state_value = 0
        for i in range(immediate_rewards.shape[0]):
            state_value += gamma ** i * immediate_rewards[i]

        return state_value, sum(immediate_rewards), step_count, original_reward


class BasicAttackerEnv(BaseAdvEnv):
    def __init__(self, config, defender_strategy: MixedDefenderStrategy):
        super().__init__(config)
        self.defender_strategy = defender_strategy
        self.done = False

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.env.num_edges, ))
        self.observation_space = gym.spaces.Box(low=0, high=100_000, shape=(self.env.num_edges, 5, ))

        # TODO fix this 5 mess
        self.defender_observation_history = deque(maxlen=5)

    def scale_action(self, action):
        positive_action = (action + 1) / 4 + 0.5
        scaled_action = np.log(positive_action / (1 - positive_action + 1e-8))
        return scaled_action

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1
        scaled_action = self.scale_action(action)
        self.defender_observation_history.append(self.env.get_travel_times_assuming_the_attack(scaled_action))

        if self.defender_strategy.predict(np.array(self.defender_observation_history), scaled_action) == True:
            state_value, cumulative_rewards, step_count, original_reward = self.get_state_values_assuming_no_action(
                self.done)
            self.done = True
            return np.ones((self.env.base.number_of_edges(), 5, )), state_value, self.done, False, {'original_reward': original_reward}
        else:
            o, r, self.done, i = self.env.step(scaled_action)
            return o, sum(r), self.done, i.get('TimeLimit.truncated', False), i

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        ObsType, dict[str, Any]]:
        self.step_count = 0
        self.defender_strategy.reset()
        self.done = False
        self.defender_observation_history.clear()
        initial_obs = self.env.reset()
        # TODO fix this 5 mess
        initial_travel_time = self.env.get_travel_times_assuming_the_attack(np.zeros((self.env.num_edges, )))
        for _ in range(5):
            self.defender_observation_history.append(initial_travel_time)

        return initial_obs, {}


class BasicDefenderEnv(BaseAdvEnv):

    # TODO Calculate precision and recall with ROC curve
    # Train a classifier defender against this mixture. assume attacker was training against a fixed budget detector.

    def __init__(self, config, attacker_strategy: MixedAttackerStrategy, penalty=1.0):
        super().__init__(config)

        self.logger = logging.getLogger(__name__)

        self.attacker_strategy = attacker_strategy
        self.previous_attacker_action = None
        self.previous_attacker_observation = None
        self.penalty = penalty

        self.observation_space = gym.spaces.Box(low=0, high=100_000, shape=(self.env.num_edges, ))
        self.action_space = gym.spaces.Discrete(2)  # 0 not raise alarm, 1 raise alarm

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1
        obs = self.env.get_travel_times_assuming_the_attack(self.previous_attacker_action)

        self.previous_attacker_observation, reward, done, info = self.env.step(self.previous_attacker_action)
        truncated = info.get('TimeLimit.truncated', False)

        # attacker_action = self.attacker_strategy.predict(self.previous_attacker_observation)
        # positive_action = (attacker_action + 1) / 4 + 0.5
        # self.previous_attacker_action = np.log(positive_action / (1 - positive_action + 1e-8))

        self.previous_attacker_action = self.attacker_strategy.predict(self.previous_attacker_observation)

        reward = -sum(reward)
        if action == 1:
            reward -= self.penalty

        # if action == 1 and not self.attacker_strategy.is_attack():
        #     self.logger.info(f'Incorrectly detected at {self.step_count}')

        # if 'correctly' detected
        if self.attacker_strategy.is_attack() and action == 1:
            state_value, cumulative_rewards, step_count, original_reward = self.get_state_values_assuming_no_action(
                done)
            done = True
            # self.logger.info(f'Correctly detected at {self.step_count}')
            return obs, -state_value, done, truncated, {'original_reward': original_reward}
        else:
            return obs, reward, done, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        ObsType, dict[str, Any]]:

        self.step_count = 0

        self.attacker_strategy.reset()

        self.previous_attacker_observation = self.env.reset()
        self.previous_attacker_observation = self.previous_attacker_observation
        self.previous_attacker_action = self.attacker_strategy.predict(self.previous_attacker_observation)
        # attacker_action = self.attacker_strategy.predict(self.previous_attacker_observation)
        # positive_action = (attacker_action + 1) / 4 + 0.5
        # self.previous_attacker_action = np.log(positive_action / (1 - positive_action + 1e-8))

        return self.env.get_travel_times_assuming_the_attack(self.previous_attacker_action), {}
