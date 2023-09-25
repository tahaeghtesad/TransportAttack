import torch.nn

from models import CustomModule


class GeneralizedAdvantageEstimation(CustomModule):
    def __init__(self, gamma, lam):
        super().__init__('GeneralizedAdvantageEstimation')
        self.gamma = gamma
        self.lam = lam

    def forward(self, values, rewards, dones, truncated):
        advantages = torch.zeros_like(values)
        last_advantage = torch.tensor(0, device=self.device)
        last_value = values[-1]

        # Calculate advantages in reverse order (from the last time step to the first)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * (1 - dones[t]) * (1 - truncated[t]) * last_value - values[t]
            last_advantage = delta + self.gamma * self.lam * (1 - dones[t]) * (1 - truncated[t]) * last_advantage
            advantages[t] = last_advantage
            last_value = values[t]

        return advantages

    def extra_repr(self) -> str:
        return f'gamma={self.gamma:.3f}, lam={self.lam:.3f}'
