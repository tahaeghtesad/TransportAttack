from abc import abstractmethod

from models import CustomModule


class ComponentInterface(CustomModule):
    @abstractmethod
    def forward(self, states, budgets, allocations, deterministic):
        raise NotImplementedError()

    @abstractmethod
    def update(self, states, actions, budgets, allocations, next_states, next_budgets, next_allocations, rewards, dones, truncateds):
        raise NotImplementedError()
