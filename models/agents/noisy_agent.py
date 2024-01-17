import torch

from models.agents.rl_agents.attackers.rl_attackers import BaseAttacker
from models.exploration import NoiseInterface


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
            noisy_budget = torch.maximum(self.budget_noise(budgets.shape), torch.zeros_like(budgets, device=self.device))
            noisy_allocation = torch.nn.functional.normalize(torch.maximum(self.allocation_noise(allocations.shape), torch.zeros_like(allocations, device=self.device)), dim=1, p=1)
            noisy_action = torch.empty(action.shape, device=self.device)
            for c in range(self.n_components):
                noisy_action[:, self.edge_component_mapping[c]] = torch.nn.functional.normalize(
                    action[:, self.edge_component_mapping[c]] + self.action_noise(action[:, self.edge_component_mapping[c]].shape), p=1, dim=1)

            return self._construct_action(noisy_action, noisy_allocation, noisy_budget), noisy_action, noisy_allocation, noisy_budget

    def _update(self, observation, allocations, budgets, action, reward, next_observation, done, truncateds):
        return self.attacker.update(observation, allocations, budgets, action, reward, next_observation, done, truncateds) | {
            'attacker/budget_noise': self.budget_noise.get_current_noise(),
            'attacker/allocation_noise': self.allocation_noise.get_current_noise(),
            'attacker/action_noise': self.action_noise.get_current_noise(),
        }
