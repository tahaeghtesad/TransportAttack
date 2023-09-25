import logging
from abc import ABC, abstractmethod

import torch.nn


class CustomModule(torch.nn.Module, ABC):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.logging = logging.getLogger(name)

    def _get_name(self):
        return f'{self.name}'

    def update(self, *args):
        raise NotImplementedError()

    @property
    def device(self) -> torch.device:
        for param in self.parameters():
            return param.device
        return torch.device('cpu')
