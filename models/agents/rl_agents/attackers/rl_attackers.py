from abc import abstractmethod, ABC

import numpy as np
import torch.nn

from models.agents import AttackerInterface, BudgetingInterface
from models.agents.heuristics.attackers.budgeting import FixedBudgeting
from models.agents.rl_agents.attackers.allocators import AllocatorInterface, NoBudgetAllocatorInterface
from models.agents.rl_agents.attackers.component import ComponentInterface
from util.scheduler import LevelTrainingScheduler, TrainingScheduler


class BaseAttacker(AttackerInterface, ABC):
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
        constructed_action, action, allocations, budgets = self.forward(observation, deterministic=deterministic)
        return constructed_action.cpu().detach().numpy()[0], action.cpu().detach().numpy()[0], allocations.cpu().detach().numpy()[0], budgets.cpu().detach().numpy()[
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

    @abstractmethod
    def _update(self, observation, allocations, budgets, action, reward, next_observation, done, truncateds):
        raise NotImplementedError('Should not be called in base class')

    def _aggregate_state(self, states):
        aggregated = torch.empty((states.shape[0], self.n_components, 5), device=self.device)
        for c in range(self.n_components):
            for f in range(5):
                aggregated[:, c, f] = torch.sum(
                    states[:, self.edge_component_mapping[c], f], dim=1
                )
        return aggregated

    def _construct_action(self, actions, allocations, budgets):
        constructed_actions = torch.zeros((actions.shape[0], self.n_edges), device=self.device)
        for c in range(self.n_components):
            constructed_actions[:, self.edge_component_mapping[c]] = actions[:, self.edge_component_mapping[c]] * allocations[:, [c]] * budgets
        return constructed_actions


class FixedBudgetNetworkedWideGreedy(BaseAttacker):

    def __init__(self, edge_component_mapping, budget) -> None:
        super().__init__(name='FixedBudgetNetworkedWideGreedy', edge_component_mapping=edge_component_mapping)
        self.budgeting = FixedBudgeting(budget)

    def forward(self, observation, deterministic):
        actions = torch.nn.functional.normalize(observation[:, :, 0], dim=1, p=1) * self.budgeting.budget
        allocations = torch.zeros((observation.shape[0], self.n_components), device=self.device)
        for c in range(self.n_components):
            allocations[:, c] = torch.sum(actions[:, self.edge_component_mapping[c]], dim=1)

        # action here is the same as constructed_action
        return actions, actions, allocations, torch.ones((observation.shape[0], 1), device=self.device) * self.budgeting.budget

    def _update(self, observation, allocations, budgets, action, reward, next_observation, done, truncateds):
        return dict()


class Attacker(BaseAttacker):
    def __init__(
            self, name, edge_component_mapping,
            budgeting: BudgetingInterface,
            allocator: AllocatorInterface,
            component: ComponentInterface,
            iterative_scheduler: TrainingScheduler,
    ):
        super().__init__(name=name, edge_component_mapping=edge_component_mapping)

        self.allocator = allocator
        self.budgeting = budgeting
        self.component = component

        self.iterative_scheduler = iterative_scheduler

    def forward(self, observation, deterministic):
        aggregated = self._aggregate_state(observation)
        budgets = self.budgeting.forward(aggregated, deterministic=False if self.iterative_scheduler.should_train('budgeting') else True)
        allocations = self.allocator.forward(aggregated, budgets, deterministic=False if self.iterative_scheduler.should_train('allocator') else True)
        actions = self.component.forward(observation, budgets, allocations, deterministic=False if self.iterative_scheduler.should_train('component') else True)
        constructed_action = self._construct_action(actions, allocations, budgets)
        return constructed_action, actions, allocations, budgets

    def _update(self, observation, allocations, budgets, actions, rewards, next_observations, dones, truncateds):
        with torch.no_grad():
            aggregated_states = self._aggregate_state(observation)
            next_aggregated = self._aggregate_state(next_observations)
            next_budgets = self.budgeting.forward(next_aggregated, deterministic=True)
            next_allocations = self.allocator.forward(next_aggregated, next_budgets,
                                                      deterministic=True)

        stats = dict()

        if self.iterative_scheduler.should_train('component'):
            stats |= self.component.update(
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

        if self.iterative_scheduler.should_train('allocator'):
            stats |= self.allocator.update(
                aggregated_states=aggregated_states,
                allocations=allocations,
                budgets=budgets,
                rewards=torch.sum(rewards, dim=1, keepdim=True),
                next_aggregated_states=next_aggregated,
                next_budgets=next_budgets,
                dones=dones,
                truncateds=truncateds,
            )

        if self.iterative_scheduler.should_train('budgeting'):
            stats |= self.budgeting.update(
                aggregated_states=aggregated_states,
                budgets=budgets,
                rewards=torch.sum(rewards, dim=1, keepdim=True),
                next_aggregated_states=next_aggregated,
                dones=dones,
                truncateds=truncateds,
            )

        return stats


class NoBudgetAttacker(BaseAttacker):
    def __init__(
            self, name, edge_component_mapping,
            allocator: NoBudgetAllocatorInterface,
            component: ComponentInterface,
            iterative_scheduler: TrainingScheduler,
    ):
        super().__init__(name=name, edge_component_mapping=edge_component_mapping)

        self.allocator = allocator
        self.component = component

        self.iterative_scheduler = iterative_scheduler

    def forward(self, observation, deterministic):
        allocations = self.allocator.forward(observation, deterministic=deterministic)
        budgets = torch.sum(allocations, dim=1, keepdim=True)
        normalized_allocations = torch.nn.functional.normalize(allocations, dim=1, p=1)
        actions = self.component.forward(observation, budgets, normalized_allocations, deterministic=deterministic)
        constructed_action = self._construct_action(actions, allocations, budgets)
        return constructed_action, actions, normalized_allocations, budgets

    def _update(self, observation, allocations, budgets, actions, rewards, next_observations, dones, truncateds):
        with torch.no_grad():
            next_allocations_times_budget = self.allocator.forward_target(next_observations, deterministic=True)
            next_allocations = torch.nn.functional.normalize(next_allocations_times_budget, dim=1, p=1)
            next_budgets = torch.sum(next_allocations_times_budget, dim=1, keepdim=True)

        stats = dict()

        if self.iterative_scheduler.should_train('component'):
            stats |= self.component.update(
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

        if self.iterative_scheduler.should_train('allocator'):
            stats |= self.allocator.update(
                states=observation,
                allocations_times_budgets=allocations * budgets,
                rewards=torch.sum(rewards, dim=1, keepdim=True),
                next_states=next_observations,
                dones=dones,
                truncateds=truncateds,
            )

        return stats
