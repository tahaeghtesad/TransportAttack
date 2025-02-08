from abc import ABC

import networkx as nx
import numpy as np
from stable_baselines3 import PPO

from transport_env.MultiAgentEnv import DynamicMultiAgentTransportationNetworkEnvironment


class BaseAttackerStrategy(ABC):

    def __init__(self):
        pass

    def predict(self, observation):
        raise NotImplementedError

    @property
    def is_attack(self) -> bool:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


# it should be a singleton
class SBAttackerStrategy(BaseAttackerStrategy):
    def __init__(self, path):
        super().__init__()
        self.policy = PPO.load(path, device='cpu')

    def predict(self, observation):

        action = self.policy.predict(observation, deterministic=False)[0]
        positive_action = (action + 1) / 4 + 0.5
        scaled_action = np.log(positive_action / (1 - positive_action + 1e-8))
        return scaled_action

    def is_attack(self):
        return True

    def reset(self):
        pass


class ZeroAttackStrategy(BaseAttackerStrategy):
    def __init__(self):
        super().__init__()

    def predict(self, observation):
        return np.zeros((observation.shape[0], ))

    def is_attack(self):
        return False

    def reset(self):
        pass


class MixedAttackerStrategy(BaseAttackerStrategy):

    def __init__(self, strategies, probabilities):
        super().__init__()
        assert len(strategies) == len(probabilities), "strategies and probabilities should have the same length"
        # assert np.sum(probabilities) == 1, "probabilities should sum to 1"
        np.testing.assert_almost_equal(np.sum(probabilities), 1.0)

        self.strategies = strategies
        self.probabilities = probabilities

        self.current_strategy: BaseAttackerStrategy = None

    def predict(self, observation):
        return self.current_strategy.predict(observation)

    def is_attack(self) -> bool:
        return self.current_strategy.is_attack()

    def reset(self) -> None:
        self.current_strategy = np.random.choice(self.strategies, p=self.probabilities)
        for strategy in self.strategies:
            strategy.reset()


class StrategicGaussianAttack(BaseAttackerStrategy):
    def __init__(self, env: DynamicMultiAgentTransportationNetworkEnvironment, index):
        super().__init__()

        assert index < len(env.edge_component_mapping)
        self.index = index
        self.attack_vector = np.zeros(len(env.base.edges))
        self.attack_vector[env.edge_component_mapping[index]] = 1

        self.capacities = np.array(list(nx.get_edge_attributes(env.base, 'capacity').values()))
        self.free_flow_times = np.array(list(nx.get_edge_attributes(env.base, 'free_flow_time').values()))
        self.b = np.array(list(nx.get_edge_attributes(env.base, 'b').values()))
        self.power = np.array(list(nx.get_edge_attributes(env.base, 'power').values()))
        self.max_vehicles = env.max_number_of_vehicles

    def get_travel_time(self, on_edge):
        on_edge = np.maximum(on_edge, 0)
        return self.free_flow_times * (1.0 + self.b * (on_edge / self.capacities) ** self.power)

    def predict(self, observation):
        values = np.random.multivariate_normal(
            mean=30 * self.capacities * self.attack_vector,
            cov=0.1 * np.eye(len(self.capacities)) * self.capacities
        )
        on_edge = observation[:, 1] * self.max_vehicles
        return np.maximum(self.get_travel_time(on_edge + values) - self.get_travel_time(on_edge), 0)

    def scale_action(self, action):
        positive_action = (action + 1) / 4 + 0.5
        scaled_action = np.log(positive_action / (1 - positive_action + 1e-8))
        return scaled_action

    def unscale(self, scaled):
        positive_action = 1 / (1 + np.exp(-scaled))
        action = 4 * (positive_action - 0.5) - 1
        one = np.float64(1)
        min_precision = np.float64(1e-16)
        return np.clip(action, -(one - min_precision), one - min_precision)

    def is_attack(self) -> bool:
        return True

    def reset(self) -> None:
        pass


class GreedyAttackStrategy(BaseAttackerStrategy):

    def __init__(self, budget: float):
        super().__init__()
        assert budget > 0, f'Budget should be positive, got {budget}'
        self.budget = budget

    def predict(self, observation):
        return observation[:, 0] / (np.linalg.norm(observation[:, 0], ord=1) + 1e-8)

    def is_attack(self) -> bool:
        return True

    def reset(self) -> None:
        pass