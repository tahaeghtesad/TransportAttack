import torch

from models.agents.rl_agents.attackers.rl_attackers import BaseAttacker
from models.exploration import NoiseInterface, EpsilonInterface


class NoisyAttacker(BaseAttacker):

    def __init__(self, attacker: BaseAttacker, budget_noise: NoiseInterface, allocation_noise: NoiseInterface, action_noise: NoiseInterface):
        super().__init__('NoisyAttacker', attacker.edge_component_mapping)
        self.attacker = attacker
        self.budget_noise = budget_noise
        self.allocation_noise = allocation_noise
        self.action_noise = action_noise

    def forward(self, observation, deterministic):
        constructed_action, action, allocations, budgets = self.attacker.forward(observation, deterministic)

        if not deterministic:
            noisy_budget = torch.maximum(budgets + self.budget_noise(budgets.shape), torch.zeros_like(budgets, device=self.device))
            noisy_allocation = torch.nn.functional.normalize(torch.maximum(allocations + self.allocation_noise(allocations.shape), torch.zeros_like(allocations, device=self.device)), dim=1, p=1)
            noisy_action = action + self.action_noise(action.shape)
            for c in range(self.n_components):
                noisy_action[:, self.edge_component_mapping[c]] = torch.nn.functional.normalize(torch.maximum(
                    noisy_action[:, self.edge_component_mapping[c]], torch.zeros_like(noisy_action[:, self.edge_component_mapping[c]], device=self.device)), p=1, dim=1)

            return self._construct_action(noisy_action, noisy_allocation, noisy_budget), noisy_action, noisy_allocation, noisy_budget

        return constructed_action, action, allocations, budgets

    def _update(self, observation, allocations, budgets, action, reward, next_observation, done, truncateds):
        return self.attacker._update(observation, allocations, budgets, action, reward, next_observation, done, truncateds) | {
            'noise/budget_noise': self.budget_noise.get_current_noise(),
            'noise/allocation_noise': self.allocation_noise.get_current_noise(),
            'noise/action_noise': self.action_noise.get_current_noise(),
        }


class EpsilonGreedyAttacker(BaseAttacker):

    def __init__(self, attacker: BaseAttacker, epsilon: EpsilonInterface) -> None:
        super().__init__('EpsilonGreedyAttacker', edge_component_mapping=attacker.edge_component_mapping)
        self.attacker = attacker
        self.epsilon = epsilon

    def forward(self, observation, deterministic):
        if not deterministic and self.epsilon():
            return self.attacker.forward(observation, deterministic)
        return self.attacker.forward(observation, True)

    def _update(self, observation, allocations, budgets, action, reward, next_observation, done, truncateds):
        return self.attacker._update(observation, allocations, budgets, action, reward, next_observation, done, truncateds) | {
            'noise/epsilon': self.epsilon.get_current_epsilon()
        }
