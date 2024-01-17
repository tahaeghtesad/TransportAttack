import torch

from models.exploration import DecayingNoiseInterface, NoiseInterface


class GaussianNoiseDecay(DecayingNoiseInterface):
    def __init__(self, start, end, decay):
        super().__init__('NoiseDecay', start, end, decay)

    def forward(self, shape):
        self.step += 1
        return torch.normal(0, self.get_current_noise(), shape, device=self.device)

    def get_current_noise(self):
        return self.end + (self.start - self.end) * torch.exp(
            -1. * self.step / self.decay)

    def extra_repr(self) -> str:
        return f'start={self.start:.2f}, end={self.end:.2f}, decay={self.decay:.2f}'


class ZeroNoise(NoiseInterface):
    def __init__(self):
        super().__init__('ZeroNoise')

    def forward(self, shape):
        return torch.zeros(shape, device=self.device)

    def get_current_noise(self):
        return torch.tensor(0.0, device=self.device)


class OUActionNoise(DecayingNoiseInterface):

    def __init__(self, mean, std_deviation, target_scale, decay, theta=0.15, dt=1e-2, x_initial=None):
        super().__init__('OUActionNoise', 1.0, target_scale, decay)
        self.theta = theta
        self.mean = torch.tensor(mean, dtype=torch.float32, device=self.device)
        self.std_dev = torch.tensor(std_deviation, dtype=torch.float32, device=self.device)
        self.dt = torch.tensor(dt)
        self.x_initial = x_initial
        self.x_prev = None
        self.target_scale = target_scale
        self.scale = self.target_scale + (1.0 - self.target_scale) * torch.exp(
            -1. * self.step / self.decay)
        self.reset()

    def forward(self, shape):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * torch.sqrt(self.dt) * torch.normal(mean=0.0, std=1.0, size=shape, device=self.device)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x

        self.step += 1
        self.scale = self.target_scale + (1.0 - self.target_scale) * torch.exp(
            -1. * self.step / self.decay)

        return x * self.scale

    def reset(self):
        super().reset()
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = torch.tensor(0.0, device=self.device)

    def get_current_noise(self):
        return self.scale * self.std_dev

    def extra_repr(self) -> str:
        return f'mean={self.mean:.2f}, std_dev={self.std_dev:.2f}, target_scale={self.target_scale:.2f}, decay={self.decay:.2f}'
