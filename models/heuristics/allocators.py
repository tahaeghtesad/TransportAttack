import torch

from models import AllocatorInterface


class ProportionalAllocator(AllocatorInterface):

    def __init__(self):
        super().__init__(name='ProportionalAllocator')

    def forward(self, aggregated_state, budgets, deterministic):
        # aggregated_state = (batch, n_components, 5)

        return torch.nn.functional.normalize(
            aggregated_state[:, :, 1], p=1, dim=1
        )

    # def get_state_dict(self):
    #     return dict()
    #
    # @classmethod
    # def from_state_dict(cls, state_dict):
    #     return cls()

    def update(self, aggregated_states, allocations, budgets, rewards, next_aggregated_states, next_budgets, dones, truncateds):
        return dict()


class SoftmaxAllocator(AllocatorInterface):

    def __init__(self):
        super().__init__(name='SoftmaxAllocator')

    def forward(self, aggregated_state, budgets, deterministic):
        # aggregated_state = (batch, n_components, 5)

        return torch.nn.functional.softmax(
            aggregated_state[:, :, 2], dim=1
        )

    # def get_state_dict(self):
    #     return dict()
    #
    # @classmethod
    # def from_state_dict(cls, state_dict):
    #     return cls()

    def update(self, aggregated_states, allocations, budgets, rewards, next_aggregated_states, next_budgets, dones):
        return dict()
