from abc import abstractmethod

import torch

import logging
from abc import ABC

import torch.nn


class CustomModule(torch.nn.Module, ABC):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.logger = logging.getLogger(name)
        self.__device: torch.device = None

    def _get_name(self):
        return f'{self.name}'

    def update(self, *args):
        raise NotImplementedError()

    @property
    def device(self) -> torch.device:
        if self.__device is not None:
            return self.__device
        for param in self.parameters():
            return param.device
        return torch.device('cpu')

    def to(self, *args, **kwargs):
        self.__device = args[0]
        for module in self.modules():
            if isinstance(module, CustomModule):
                if module.__device is None:
                    module.to(*args, **kwargs)
        return super().to(*args, **kwargs)


class DetectorInterface(CustomModule):

    @abstractmethod
    def __init__(self, name):
        super().__init__(name)

    @abstractmethod
    def forward(self, edge_travel_times, deterministic):
        raise NotImplementedError()

    @abstractmethod
    def update(self, edge_travel_times, decision, next_edge_travel_times, rewards, dones):
        raise NotImplementedError()
