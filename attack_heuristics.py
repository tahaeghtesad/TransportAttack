import logging
from typing import List

import gym
import networkx as nx

import numpy as np


def get_path_travel_time(path: List[int], weight):
    distance = 0
    for i in range(len(path) - 1):
        distance += weight(path[i], path[i + 1], None)
    return distance


def estimate_delay_for_action(
        action,
        network_graph: gym.spaces.GraphInstance,
        decision_graph: gym.spaces.GraphInstance
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

        assert perturbed_distance >= correct_distance,\
            f'Perturbed path is shorter than correct path'
        delay += perturbed_distance - correct_distance

    return delay


class BaseHeuristic:
    def __init__(self, action_space, name):
        self.action_space = action_space
        self.logger = logging.getLogger(name)

        self.name = name
        self.step = 0

    def predict(self, obs):
        self.step += 1

    def reset(self):
        self.step = 0


class PostProcessHeuristic:
    def __init__(self, heuristic: BaseHeuristic):
        self.heuristic = heuristic
        self.logger = logging.getLogger(self.heuristic.name)
        self.name = self.heuristic.name

    def predict(self, obs):
        self.logger.debug(f'Step {self.heuristic.step} - Obs: {obs}')
        action = self.heuristic.predict(obs)
        self.logger.debug(f'Step {self.heuristic.step} - Action: {action}')
        return action


class Zero(BaseHeuristic):
    def __init__(self, action_space):
        super().__init__(action_space, self.__class__.__name__)

    def predict(self, obs):
        super().predict(obs)
        action = np.zeros(self.action_space.shape)
        return action


class Random(BaseHeuristic):
    def __init__(self, action_space,
                 norm,
                 epsilon,
                 frac,  # In case 'selection' is 'continuous' this does not matter
                 selection  # Can be either 'continuous' or 'discrete'
                 ):
        super().__init__(action_space, f'{self.__class__.__name__}.{selection}')

        assert 0 <= frac <= 1, f'Invalid fraction, {frac}'

        self.selection = selection
        self.norm = norm
        self.epsilon = epsilon
        self.frac = frac

    def predict(self, obs):
        super().predict(obs)

        action = np.zeros(self.action_space.shape)
        if self.selection == 'continuous':
            action = np.random.rand(
                *self.action_space.shape
            )
        elif self.selection == 'discrete':
            action = self.epsilon * np.random.choice(
                [0, 1],
                size=self.action_space.shape,
                p=[1 - self.frac, self.frac]
            )
        else:
            raise Exception(f'Invalid "selection" criteria: {self.selection}')

        return action * self.epsilon / np.linalg.norm(action, self.norm)


class MultiRandom(BaseHeuristic):
    def __init__(self,
                 action_space,
                 num_sample,
                 action_type,
                 frac,
                 norm,
                 epsilon):
        super().__init__(action_space, f'{self.__class__.__name__}.{action_type}')

        assert action_type in ['continuous', 'discrete'], \
            f'Invalid "Action Type" {action_type}'
        assert 0 <= frac <= 1, f'invalid fraction: {frac:.2f}'

        self.generator = Random(action_space, norm, epsilon, frac, action_type)

        self.num_sample = num_sample

    def predict(self, obs):
        nodes, edges, edge_links = obs[1]
        if nodes[:, 0].sum() == 0:
            return np.zeros(self.action_space.shape)

        actions = [
            self.generator.predict(obs) for _ in range(self.num_sample)
        ]

        max_index = 0
        max_perturbed = 0
        self.logger.debug(f'Suggestion Perturbations: ')
        for i, action in enumerate(actions):
            self.logger.debug(f'Perturbation {i} - {action}')
            if delay := estimate_delay_for_action(action, *obs) > max_perturbed:
                max_perturbed = delay
                max_index = i
        return actions[max_index]


class GreedyRider(BaseHeuristic):

    def __init__(self, action_space, epsilon, norm):
        super().__init__(action_space, self.__class__.__name__)
        self.norm = norm
        self.epsilon = epsilon

    def predict(self, obs):
        super().predict(obs)
        network_graph: gym.spaces.GraphInstance = obs[0]
        decision_graph: gym.spaces.GraphInstance = obs[1]

        action = np.zeros(self.action_space.shape)

        if decision_graph.nodes.sum() == 0:
            return action

        reconstructed_graph: nx.DiGraph = nx.from_edgelist(
            obs[0].edge_links,
            create_using=nx.DiGraph
        )
        on_edge = dict()
        increase = dict()
        for i, (e, info) in enumerate(zip(
                reconstructed_graph.edges,
                network_graph.edges)):
            on_edge[e] = info[1]
        for source, dest in decision_graph.edge_links:
            path = nx.shortest_path(reconstructed_graph, source, dest, weight=lambda u, v, _: on_edge[e])
            first_edge = path[0], path[1]
            if first_edge in increase:
                increase[first_edge] += 1
            else:
                increase[first_edge] = 1

        for i, e in enumerate(reconstructed_graph.edges):
            action[i] = increase[e] if e in increase else 0

        normalized_action = action * self.epsilon / np.linalg.norm(action, self.norm)

        return normalized_action
