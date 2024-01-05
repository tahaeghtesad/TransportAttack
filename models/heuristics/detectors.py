import torch

from models.dl.detectors import BaseDetector


class ZeroDetector(BaseDetector):

    def __init__(self):
        super().__init__('ZeroDetector')

    def forward(self, edge_travel_times, deterministic):
        return torch.unsqueeze(torch.zeros(edge_travel_times.shape[0], device=edge_travel_times.device), dim=1)

    def _update(self, edge_travel_times, decisions, next_edge_travel_times, rewards, dones):
        return dict()
