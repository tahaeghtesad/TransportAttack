import logging
from abc import abstractmethod
from typing import List

import gym
import networkx as nx

import numpy as np
import torch

from models import CustomModule
from models.rl_attackers import BaseAttacker
from util.visualize import Timer


def get_path_travel_time(path: List[int], weight):
    distance = 0
    for i in range(len(path) - 1):
        distance += weight(path[i], path[i + 1], None)
    return distance


def estimate_delay_for_action(
        action,
        network_graph,
        decision_graph
):
    reconstructed_graph: nx.DiGraph = nx.from_edgelist(
        network_graph.edge_links,
        create_using=nx.DiGraph
    )

    perturbed = dict()
    on_edge = dict()
    for i, (e, info) in enumerate(zip(reconstructed_graph.edges, network_graph.edges)):
        perturbed[e] = action[i]
        on_edge[e] = info[1]

    delay = 0

    for t in decision_graph.edge_links:
        perturbed_path = nx.shortest_path(
            reconstructed_graph,
            t[0],
            t[1],
            weight=lambda u, v, _: perturbed[(u, v)] + on_edge[(u, v)]
        )
        correct_path = nx.shortest_path(
            reconstructed_graph,
            t[0],
            t[1],
            weight=lambda u, v, _: on_edge[(u, v)]
        )

        perturbed_distance = get_path_travel_time(perturbed_path, lambda u, v, _: on_edge[(u, v)])
        correct_distance = get_path_travel_time(correct_path, lambda u, v, _: on_edge[(u, v)])

        assert perturbed_distance >= correct_distance, \
            f'Perturbed path is shorter than correct path'
        delay += perturbed_distance - correct_distance

    return delay


class BaseHeuristic(BaseAttacker):
    def __init__(self, name, edge_component_mapping):
        super().__init__(name, edge_component_mapping=edge_component_mapping)

    @abstractmethod
    def forward(self, observation, deterministic):
        raise NotImplementedError()


class Zero(BaseHeuristic):

    def __init__(self, edge_component_mapping):
        super().__init__('Zero', edge_component_mapping=edge_component_mapping)
        self.n_edges = sum([len(c) for c in edge_component_mapping])

    def forward(self, observation, deterministic):
        n_batch = observation.shape[0]
        # should return action, allocation, budgets
        return (
            torch.zeros((n_batch, self.n_edges), device=self.device),
            torch.zeros((n_batch, self.n_components), device=self.device),
            torch.zeros((n_batch, 1), device=self.device)
        )

# TODO fix these
# class Random(BaseHeuristic):
#     def __init__(self, action_shape,
#                  norm,
#                  epsilon,
#                  frac,  # In case 'selection' is 'continuous' this does not matter
#                  selection  # Can be either 'continuous' or 'discrete'
#                  ):
#         super().__init__(action_shape, f'{self.__class__.__name__}.{selection}')
#
#         assert 0 <= frac <= 1, f'Invalid fraction, {frac}'
#
#         self.selection = selection
#         self.norm = norm
#         self.epsilon = epsilon
#         self.frac = frac
#
#     def predict(self, obs):
#         super().predict(obs)
#
#         action = np.zeros(self.action_shape)
#         if self.selection == 'continuous':
#             action = np.random.rand(
#                 *self.action_shape
#             )
#         elif self.selection == 'discrete':
#             action = self.epsilon * np.random.choice(
#                 [0, 1],
#                 size=self.action_shape,
#                 p=[1 - self.frac, self.frac]
#             )
#         else:
#             raise Exception(f'Invalid "selection" criteria: {self.selection}')
#
#         norm = np.linalg.norm(action, self.norm)
#         if norm == 0:
#             self.logger.warning(f'Random {self.selection} action is zero')
#
#         return self.epsilon * np.divide(action, norm, where=norm != 0)
#
#
# class MultiRandom(BaseHeuristic):
#     def __init__(self,
#                  action_space,
#                  num_sample,
#                  action_type,
#                  frac,
#                  norm,
#                  epsilon):
#         super().__init__(action_space, f'{self.__class__.__name__}.{action_type}')
#
#         assert action_type in ['continuous', 'discrete'], \
#             f'Invalid "Action Type" {action_type}'
#         assert 0 <= frac <= 1, f'invalid fraction: {frac:.2f}'
#
#         self.generator = Random(action_space, norm, epsilon, frac, action_type)
#
#         self.num_sample = num_sample
#
#     def predict(self, obs):
#         nodes, edges, edge_links = obs[1]
#         if nodes[:, 0].sum() == 0:
#             return np.zeros(self.action_shape)
#
#         actions = [
#             self.generator.predict(obs) for _ in range(self.num_sample)
#         ]
#
#         max_index = 0
#         max_perturbed = 0
#         self.logger.log(1, f'Suggestion Perturbations: ')
#         for i, action in enumerate(actions):
#             self.logger.log(1, f'Perturbation {i} - {action}')
#             if delay := estimate_delay_for_action(action, *obs) > max_perturbed:
#                 max_perturbed = delay
#                 max_index = i
#         return actions[max_index]


