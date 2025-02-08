from collections import deque
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, ActType


class HistoryEnvironment(gym.Wrapper):

    def __init__(self, base: gym.Env, n_history: int):
        super().__init__(base)
        self.n_history = n_history
        self.history = deque(maxlen=n_history)

        self.observation_space = gym.spaces.Box(
            shape=(n_history, ) + self.env.observation_space.shape,
            low=self.env.observation_space.low.min(),
            high=self.env.observation_space.high.max()
        )
        self.action_space = self.env.action_space

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, done, truncated, info = self.env.step(action)
        self.history.append(obs)
        return np.array(self.history), reward, done, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        ObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.history.clear()
        for _ in range(self.n_history):
            self.history.append(obs)
        return np.array(self.history), info