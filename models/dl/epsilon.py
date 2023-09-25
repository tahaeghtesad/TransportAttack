import torch

from models import EpsilonInterface


class DecayEpsilon(EpsilonInterface):
    def __init__(self, epsilon_start, epsilon_end, epsilon_decay):
        super().__init__('DecayEpsilon')
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = torch.tensor(epsilon_decay, device=self.device)
        self.step = 0

    def reset(self):
        super().reset()

    def forward(self):
        epsilon = self.get_current_epsilon()
        self.step += 1
        return torch.rand(1) < epsilon

    def get_current_epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * torch.exp(
            -1. * self.step / self.epsilon_decay)


class ConstantEpsilon(EpsilonInterface):
    def __init__(self, epsilon):
        super().__init__('ConstantEpsilon')
        self.epsilon = torch.tensor(epsilon, device=self.device)

    def forward(self):
        return torch.rand(1) < self.epsilon

    def get_current_epsilon(self):
        return self.epsilon


class NoEpsilon(EpsilonInterface):
    def __init__(self):
        super().__init__('NoEpsilon')

    def forward(self):
        return False

    def get_current_epsilon(self):
        return torch.tensor(0.0, device=self.device)
