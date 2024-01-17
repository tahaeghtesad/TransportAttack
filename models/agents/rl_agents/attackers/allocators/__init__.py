from abc import abstractmethod

from models import CustomModule


class AllocatorInterface(CustomModule):

    @abstractmethod
    def __init__(self, name):
        super().__init__(name)

    @abstractmethod
    def forward(self, aggregated_state, budgets, deterministic):
        raise NotImplementedError()

    @abstractmethod
    def update(self, aggregated_states, allocations, budgets, rewards, next_aggregated_states, next_budgets, dones, truncateds):
        raise NotImplementedError()


class NoBudgetAllocatorInterface(CustomModule):

    @abstractmethod
    def __init__(self, name):
        super().__init__(name)

    @abstractmethod
    def forward(self, aggregated_state, deterministic):
        raise NotImplementedError()

    @abstractmethod
    def update(self, aggregated_states, allocations, rewards, next_aggregated_states, dones, truncateds):
        raise NotImplementedError()
