import torch

from models.agents.rl_agents.defenders.detectors.detectors import BaseDetector
from models.exploration import EpsilonInterface


class EpsilonGreedyDetector(BaseDetector):

    def __init__(self, agent: BaseDetector, epsilon: EpsilonInterface):
        super().__init__('EpsilonGreedyDetector')
        self.epsilon = epsilon
        self.agent = agent

    def forward(self, edge_travel_times, deterministic):
        original_action = self.agent.forward(edge_travel_times, deterministic)
        if not deterministic and self.epsilon():
            return torch.randint_like(original_action, low=0, high=2)
        else:
            return original_action

    def _update(self, edge_travel_times, decisions, next_edge_travel_times, rewards, dones):
        return self.agent._update(edge_travel_times, decisions, next_edge_travel_times, rewards, dones) | {
            'detector/epsilon': self.epsilon.get_current_epsilon()
        }
