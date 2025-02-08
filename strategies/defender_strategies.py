import logging
import random
from abc import ABC
from typing import Union

import numpy as np
from scipy.stats import multivariate_normal
from stable_baselines3 import PPO

from util.math import sigmoid


class BaseDefenderStrategy(ABC):
    def __init__(self):
        pass

    def predict(self, observation, attacker_action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class NoDefense(BaseDefenderStrategy):
    def __init__(self):
        super().__init__()

    def predict(self, observation, attacker_action):
        return False

    def reset(self):
        pass


class DeterministicBudgetDefender(BaseDefenderStrategy):
    def __init__(self, budget):
        super().__init__()
        self.budget = budget

    def predict(self, observation, attacker_action):
        if sum(attacker_action) > self.budget:
            return True
        return False

    def reset(self):
        pass


class StochasticBudgetDefender(BaseDefenderStrategy):
    def __init__(self, budget):
        super().__init__()
        self.budget = budget
        self.logger = logging.getLogger(__name__)

    def predict(self, observation, attacker_action):
        attacker_budget = sum(attacker_action)
        detected = sigmoid(attacker_budget, 1.5, self.budget) > random.random()
        return detected

    def reset(self):
        pass


class SBDefenderStrategy(BaseDefenderStrategy):
    def __init__(self, path):
        super().__init__()
        self.policy = PPO.load(path, device='cpu')

    def predict(self, observation, attacker_action):
        return self.policy.predict(observation, deterministic=True)[0]

    def reset(self):
        pass


class MixedDefenderStrategy(BaseDefenderStrategy):
    def __init__(self, policies: list[BaseDefenderStrategy], probabilities):
        super().__init__()
        self.policies = policies
        self.probabilities = probabilities

        self.current_policy: BaseDefenderStrategy = None

    def reset(self):
        self.current_policy = random.choices(self.policies, self.probabilities)[0]
        for policy in self.policies:
            policy.reset()

    def predict(self, observation, attacker_action):
        return self.current_policy.predict(observation, attacker_action)


class BayesianDetector(BaseDefenderStrategy):

    def __init__(self, normal_mean, normal_cov, threshold=0.8):  # Threshold is the confidence that the observation is attack
        super().__init__()
        self.observation_list = []
        self.normal_mean = normal_mean
        self.normal_cov = normal_cov
        self.threshold = threshold
        self.mvn = multivariate_normal(normal_mean.flatten(), normal_cov, allow_singular=True)

    @classmethod
    def from_normal_observation(cls, normal_observations: Union[np.ndarray, str], threshold=0.8):
        if isinstance(normal_observations, str):
            normal_observations = np.load(normal_observations)

        normal_observations = normal_observations[:, :, 0, :]
        mean = np.mean(normal_observations, axis=0)
        cov = np.cov(normal_observations.reshape(normal_observations.shape[0], -1), rowvar=False)
        return cls(mean, cov, threshold)

    def predict(self, observation, attacker_action):
        if len(self.observation_list) < 5:
            self.observation_list.append(observation[-1, :])
            return False
        else:
            return self.mvn.pdf(np.array(self.observation_list).flatten()) < (1 - self.threshold)

    def reset(self):
        self.observation_list = []