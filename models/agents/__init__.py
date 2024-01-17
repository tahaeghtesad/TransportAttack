from abc import abstractmethod

from models import CustomModule


class BudgetingInterface(CustomModule):

    @abstractmethod
    def forward(self, aggregated_state, deterministic):
        raise NotImplementedError()

    @abstractmethod
    def update(self, aggregated_states, budgets, rewards, next_aggregated_states, dones, truncateds):
        raise NotImplementedError()


class AttackerInterface(CustomModule):
    @abstractmethod
    def forward(self, observation, deterministic):
        raise NotImplementedError()

    @abstractmethod
    def update(self, observation, allocations, budgets, action, reward, next_observation, done, truncateds):
        raise NotImplementedError()