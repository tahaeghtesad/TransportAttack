from abc import abstractmethod

import numpy as np
import torch

from models import DetectorInterface, CustomModule
from util.torch.math import r2_score
from util.torch.misc import hard_sync, soft_sync


class QCritic(CustomModule):
    def __init__(self, n_features, n_edges, lr):
        super().__init__('QCritic')

        self.n_features = n_features
        self.n_edges = n_edges
        self.lr = lr

        self.state_extractor = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(n_features * n_edges, 512),
            torch.nn.ReLU(),
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def extra_repr(self) -> str:
        return f'n_features={self.n_features}, n_edges={self.n_edges}, lr={self.lr}, Action Space={{True/1, False/1}}'

    def forward(self, observation):
        return self.model(
            self.state_extractor(observation)
        )


class BaseDetector(DetectorInterface):
    def __init__(self, name):
        super(DetectorInterface, self).__init__(name)

    def update(self, edge_travel_times, decisions, next_edge_travel_times, rewards, dones):
        return self._update(
            torch.from_numpy(np.array(edge_travel_times)).float().to(self.device),
            torch.from_numpy(np.array(decisions)).float().to(self.device),
            torch.from_numpy(np.array(next_edge_travel_times)).float().to(self.device),
            torch.from_numpy(np.array(rewards)).float().to(self.device),
            torch.from_numpy(np.array(dones)).float().to(self.device)
        )

    def forward_single(self, edge_travel_time, deterministic):
        x = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(edge_travel_time), dim=0), dim=2).float().to(self.device)
        return bool(self.forward(x, deterministic).cpu().data.numpy()[0][0].item())

    @abstractmethod
    def forward(self, edge_travel_times, deterministic):
        raise NotImplementedError

    @abstractmethod
    def _update(self, edge_travel_times, decisions, next_edge_travel_times, rewards, dones):
        raise NotImplementedError('Should not be called in base class')


class DoubleDQNDetector(BaseDetector):
    def __init__(self, n_edges, n_features, gamma, tau, lr):
        super(DetectorInterface, self).__init__('DoubleDQNDetector')

        self.n_edges = n_edges
        self.n_features = n_features
        self.lr = lr
        self.gamma = gamma
        self.tau = tau

        self.model = QCritic(n_features, n_edges, lr)
        self.target_model = QCritic(n_features, n_edges, lr)

        hard_sync(self.target_model, self.model)

    def forward(self, edge_travel_times, deterministic=True):
        return torch.argmax(self.model.forward(edge_travel_times), dim=1, keepdim=True)

    def _update(self, edge_travel_times, decisions, next_edge_travel_times, rewards, dones):
        with torch.no_grad():
            next_q_values = torch.max(self.target_model.forward(next_edge_travel_times), dim=1, keepdim=True)[0]  # torch.max() returns a tuple (values, indices)
            target_q_values = torch.unsqueeze(rewards, dim=1) + (1 - torch.unsqueeze(dones, dim=1)) * self.gamma * next_q_values

        q_values = self.model.forward(edge_travel_times).gather(dim=1, index=torch.unsqueeze(decisions, dim=1).to(torch.int64))
        loss = torch.nn.functional.mse_loss(q_values, target_q_values)

        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

        soft_sync(self.target_model, self.model, self.tau)

        return {
            'detector/q_loss': loss.cpu().detach().numpy().item(),
            'detector/r2': r2_score(target_q_values, q_values).cpu().detach().numpy().item(),
            'detector/max_q': q_values.max().cpu().detach().numpy().item(),
            'detector/mean_q': q_values.mean().cpu().detach().numpy().item(),
            'detector/min_q': q_values.min().cpu().detach().numpy().item(),
        }
