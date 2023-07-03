import logging

import numpy as np


class BudgetAgent:
    def __init__(self, env) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.env = env

        self.n_components = self.env.config['n_components']
        self.budget = self.env.config['epsilon']
        self.norm = self.env.config['norm']

    def predict(self, on_vertex):
        allocation = np.zeros(self.n_components)
        for n in self.env.base.nodes:
            allocation[self.env.base.nodes[n]['component']] += on_vertex[n]
        allocation_norm = np.linalg.norm(allocation, ord=self.norm)
        allocation = np.divide(allocation, allocation_norm, where=allocation_norm != 0) * self.budget
        self.logger.info(f'Allocation: {allocation}')
        return allocation
