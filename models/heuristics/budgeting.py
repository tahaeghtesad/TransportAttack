import torch

from models import BudgetingInterface
from models.dl.noise import ZeroNoise


class FixedBudgeting(BudgetingInterface):
    def __init__(self, budget, noise=ZeroNoise()):
        super().__init__(name='FixedBudgeting')
        self.budget = torch.nn.Parameter(torch.tensor(budget), requires_grad=False)
        self.noise = noise

    def forward(self, aggregated_state, deterministic):
        budget = torch.ones(aggregated_state.shape[0], 1, device=self.device) * self.budget * torch.gt(torch.sum(aggregated_state[:, :, 1], dim=1, keepdim=True), 0)
        if not deterministic:
            return torch.maximum(budget + self.noise(budget.shape), torch.zeros_like(budget, device=self.device))
        else:
            return budget

    def update(self, aggregated_state, budget, reward, next_aggregated_state, done, truncateds):
        return {
            'budgeting/noise': self.noise.get_current_noise().detach().cpu().numpy().item(),
        }

    def extra_repr(self) -> str:
        return f'budget={self.budget.detach().cpu().numpy().item():.2f}'

    # def get_state_dict(self):
    #     return dict(
    #         budget=self.budget.detach().cpu().numpy().item(),
    #         noise=self.noise.get_state_dict()
    #     )
    #
    # @classmethod
    # def from_state_dict(cls, state_dict):
    #     return cls(
    #         budget=state_dict['budget'],
    #         noise=OUActionNoise.from_state_dict(state_dict['noise'])
    #     )
