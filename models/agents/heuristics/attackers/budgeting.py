import torch

from models.agents import BudgetingInterface


class FixedBudgeting(BudgetingInterface):
    def __init__(self, budget):
        super().__init__(name='FixedBudgeting')
        self.budget = torch.nn.Parameter(torch.tensor(budget), requires_grad=False)

    def forward(self, aggregated_state, deterministic):
        return torch.ones(aggregated_state.shape[0], 1, device=self.device) * self.budget

    def update(self, aggregated_states, budgets, rewards, next_aggregated_states, dones, truncateds):
        return {}

    def extra_repr(self) -> str:
        return f'budget={self.budget.detach().cpu().numpy().item():.2f}'
