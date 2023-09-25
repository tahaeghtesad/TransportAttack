from abc import abstractmethod

import numpy as np
import torch.nn

from models import AttackerInterface, BudgetingInterface, AllocatorInterface, ComponentInterface
from models.heuristics.budgeting import FixedBudgeting


class BaseAttacker(AttackerInterface):
    def __init__(self, name, edge_component_mapping) -> None:
        super().__init__(name)
        self.edge_component_mapping = edge_component_mapping
        self.n_components = len(edge_component_mapping)
        self.n_edges = sum([len(v) for v in edge_component_mapping])

    @abstractmethod
    def forward(self, observation, deterministic):
        raise NotImplementedError

    def forward_single(self, observation, deterministic):
        observation = torch.unsqueeze(torch.from_numpy(observation), dim=0).to(self.device)
        action, allocations, budgets = self.forward(observation, deterministic=deterministic)
        return action.cpu().detach().numpy()[0], allocations.cpu().detach().numpy()[0], budgets.cpu().detach().numpy()[
            0]

    def update(self, observation, allocations, budgets, action, reward, next_observation, done, truncateds):
        observation = torch.from_numpy(np.array(observation)).float().to(self.device)
        allocations = torch.from_numpy(np.array(allocations)).float().to(self.device)
        budgets = torch.from_numpy(np.array(budgets)).float().to(self.device)
        action = torch.from_numpy(np.array(action)).float().to(self.device)
        reward = torch.from_numpy(np.array(reward)).float().to(self.device)
        next_observation = torch.from_numpy(np.array(next_observation)).float().to(self.device)
        done = torch.from_numpy(np.array(done)).float().to(self.device).unsqueeze(dim=1)
        truncateds = torch.from_numpy(np.array(truncateds)).float().to(self.device).unsqueeze(dim=1)
        return self._update(observation, allocations, budgets, action, reward, next_observation, done, truncateds)

    def _update(self, observation, allocations, budgets, action, reward, next_observation, done, truncateds):
        raise NotImplementedError('Should not be called in base class')

    def _aggregate_state(self, states):
        aggregated = torch.empty((states.shape[0], self.n_components, 5), device=self.device)
        for c in range(self.n_components):
            aggregated[:, c, :] = torch.sum(
                states[:, self.edge_component_mapping[c]], dim=1
            )
        return aggregated


class FixedBudgetNetworkedWideGreedy(BaseAttacker):
    def __init__(self, edge_component_mapping, budget, budget_noise) -> None:
        super().__init__(name='FixedBudgetNetworkedWideGreedy', edge_component_mapping=edge_component_mapping)
        self.budgeting = FixedBudgeting(budget, budget_noise)

    def forward(self, observation, deterministic):
        actions = torch.nn.functional.normalize(observation[:, :, 0], dim=1, p=1) * self.budgeting.budget
        allocations = torch.zeros((observation.shape[0], self.n_components), device=self.device)
        for c in range(self.n_components):
            allocations[:, c] = torch.sum(actions[:, self.edge_component_mapping[c]], dim=1)
        return actions, allocations, torch.ones((observation.shape[0], 1), device=self.device) * self.budgeting.budget


class Attacker(BaseAttacker):
    def __init__(
            self, name, edge_component_mapping,
            budgeting: BudgetingInterface,
            allocator: AllocatorInterface,
            component: ComponentInterface,
    ):
        super().__init__(name=name, edge_component_mapping=edge_component_mapping)

        self.allocator = allocator
        self.budgeting = budgeting
        self.component = component

    def forward(self, observation, deterministic):
        aggregated = self._aggregate_state(observation)
        budgets = self.budgeting.forward(aggregated, deterministic=deterministic)
        allocations = self.allocator.forward(aggregated, budgets, deterministic=deterministic)
        actions = self.component.forward(observation, budgets, allocations, deterministic=deterministic)
        return actions, allocations, budgets

    def _update(self, observation, allocations, budgets, actions, rewards, next_observations, dones, truncateds):
        with torch.no_grad():
            next_aggregated = self._aggregate_state(next_observations)
            next_budgets = self.budgeting.forward(next_aggregated, deterministic=True)  # TODO is this True?
            next_allocations = self.allocator.forward(next_aggregated, next_budgets,
                                                      deterministic=True)  # TODO is this True?

        attacker_stat = self.component.update(
            states=observation,
            actions=actions,
            budgets=budgets,
            allocations=allocations,
            next_states=next_observations,
            next_budgets=next_budgets,
            next_allocations=next_allocations,
            rewards=rewards,
            dones=dones,
            truncateds=truncateds
        )

        allocator_stat = self.allocator.update(
            aggregated_states=next_aggregated,
            allocations=allocations,
            budgets=budgets,
            rewards=torch.sum(rewards, dim=1, keepdim=True),
            next_aggregated_states=next_aggregated,
            next_budgets=next_budgets,
            dones=dones,
            truncateds=truncateds,
        )

        budgeting_stat = self.budgeting.update(
            aggregated_state=next_aggregated,
            budget=next_budgets,
            reward=torch.sum(rewards, dim=1, keepdim=True),
            next_aggregated_state=next_aggregated,
            done=dones,
            truncateds=truncateds,
        )

        return attacker_stat | allocator_stat | budgeting_stat