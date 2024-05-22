from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, ActType

from transport_env.MultiAgentEnv import DynamicMultiAgentTransportationNetworkEnvironment


class GymMultiEnv(gym.Env):
    def __init__(self, config):
        self.env = DynamicMultiAgentTransportationNetworkEnvironment(config)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.env.num_edges, ))
        self.observation_space = gym.spaces.Box(low=0, high=1_000_000, shape=(self.env.num_edges * 5, ))

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action += 1
        action = action / action.sum() * 30.0
        o, r, d, i = self.env.step(action)
        return o.flatten(), r.sum(), d, i.get('TimeLimit.truncated', False), i

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        ObsType, dict[str, Any]]:
        return self.env.reset().flatten(), {}

    def get_adjacency_matrix(self):
        return self.env.get_adjacency_matrix()


