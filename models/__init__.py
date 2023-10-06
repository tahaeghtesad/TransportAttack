from abc import abstractmethod

import torch

from models.dl.util import CustomModule


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


class ComponentInterface(CustomModule):
    @abstractmethod
    def forward(self, states, budgets, allocations, deterministic):
        raise NotImplementedError()

    @abstractmethod
    def update(self, states, actions, budgets, allocations, next_states, next_budgets, next_allocations, rewards, dones, truncateds):
        raise NotImplementedError()


class BudgetingInterface(CustomModule):

    @abstractmethod
    def forward(self, aggregated_state, deterministic):
        raise NotImplementedError()

    @abstractmethod
    def update(self, aggregated_state, budget, reward, next_aggregated_state, done, truncateds):
        raise NotImplementedError()


class AttackerInterface(CustomModule):
    @abstractmethod
    def forward(self, observation, deterministic):
        raise NotImplementedError()

    @abstractmethod
    def update(self, observation, allocations, budgets, action, reward, next_observation, done, truncateds):
        raise NotImplementedError()


class NoiseInterface(CustomModule):

    @abstractmethod
    def __init__(self, name):
        super().__init__(name)

    @abstractmethod
    def forward(self, shape):
        raise NotImplementedError()

    @abstractmethod
    def get_current_noise(self):
        raise NotImplementedError()


class DecayingNoiseInterface(NoiseInterface):

    @abstractmethod
    def __init__(self, name, start, end, decay):
        super().__init__(name)
        self.start = torch.tensor(start, dtype=torch.float32, device=self.device)
        self.end = torch.tensor(end, dtype=torch.float32, device=self.device)
        self.decay = torch.tensor(decay, dtype=torch.float32, device=self.device)
        self.step = 0

    def reset(self):
        self.step = 0


class EpsilonInterface(CustomModule):

    @abstractmethod
    def __init__(self, name):
        super().__init__(name)

    @abstractmethod
    def forward(self):
        raise NotImplementedError()

    @abstractmethod
    def get_current_epsilon(self):
        raise NotImplementedError()

