from abc import abstractmethod

import torch

from models import CustomModule


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