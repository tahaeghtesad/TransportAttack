import logging
import random

import gym
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from transport_env.BaseNetworkEnv import BaseTransportationNetworkEnv
from util.math import sigmoid


class DynamicMultiAgentTransportationNetworkEnvironment(BaseTransportationNetworkEnv):

    def __init__(self, config, base_path='.') -> None:
        super().__init__(config, base_path)
        self.logger = logging.getLogger(__name__)

        self.n_components = config['n_components']
        self.partition_graph(self.n_components, n_repeat=500)
        self.edge_component_mapping = self.__get_edge_component_mapping()

        self.logger.info(f'Components: {[len(self.edge_component_mapping[comp]) for comp in range(self.n_components)]}')

        self.action_space = [
            gym.spaces.Box(low=0.0, high=1.0, shape=(len(self.edge_component_mapping[comp]),)) for comp in range(self.n_components)
        ]

        self.observation_space = [
            gym.spaces.Box(low=0.0, high=np.inf, shape=(len(self.edge_component_mapping[comp]), 5)) for comp in range(self.n_components)
        ]  # Observation: dict(feature_vector, allocation)

        vehicles_in_components = [[0, 0] for _ in range(self.n_components)]
        for t in self.base_trips:
            vehicles_in_components[self.base.nodes[t.next_node]['component']][0] += t.count
            vehicles_in_components[self.base.nodes[t.next_node]['component']][1] += 1

        self.logger.info(f'Vehicles in component: {vehicles_in_components}')

    def reset(self):
        super().reset()

        return self.get_current_observation_edge_vector()

    def step(self, action: np.ndarray):
        original_action = action
        action = np.maximum(action, 0)

        if not self.initialized:
            raise Exception('Call env.reset() to initialize the network and trips before env.step().')
        if self.finished:
            raise Exception('Previous epoch has ended. Call env.reset() ro reinitialize the network.')

        self.time_step += 1

        remaining_trips = 0
        on_edge = self.get_on_edge_vehicles()
        on_vertex = self.get_on_vertex_vehicles()

        # # Calculating component action allocation penalty
        # component_allocation = np.zeros(self.n_components)
        # for c in range(self.n_components):
        #     component_allocation[c] = np.sum(action[self.edge_component_mapping[c]])
        # component_allocation_norm = np.linalg.norm(component_allocation, ord=self.norm)
        # component_allocation = np.divide(component_allocation, component_allocation_norm, where=component_allocation_norm != 0)
        # component_action_penalty = np.maximum(0, component_allocation - self.last_allocation)
        #
        # # Normalizing to component allocation
        # for c in range(self.n_components):
        #     component_norm = np.linalg.norm(action[self.edge_component_mapping[c]], ord=self.norm)
        #     action[self.edge_component_mapping[c]] = np.divide(action[self.edge_component_mapping[c]], component_norm, where=component_norm != 0) * self.allocation * self.last_allocation[c]

        # Calculating the transition

        perturbed = dict()
        for i, e in enumerate(self.base.edges):
            perturbed[e] = action[i]

        currently_finished = 0
        component_time_diff = np.zeros(self.n_components)
        arrived_vehicles = np.zeros(self.n_components)

        for trip in self.trips:
            if trip.next_node == trip.destination and trip.time_to_next == 0:  # trip is finished
                self.logger.debug(f'Step: {self.time_step}. Trip {trip.number} finished.')
                currently_finished += 1
                continue

            # trip progressed for 1 step
            remaining_trips += 1
            trip.progress += 1

            # vehicle is on a node, we will calculate the next node and send the vehicle on that destination
            if trip.time_to_next == 0:
                path = nx.shortest_path(
                    self.base,
                    trip.next_node,
                    trip.destination,
                    weight=lambda u, v, _: np.maximum(0, self.get_travel_time(u, v, on_edge[(u, v)]) + perturbed[(u, v)])
                )
                path_weights = [self.get_travel_time(path[i], path[i+1], on_edge[(path[i], path[i+1])]) + perturbed[(path[i], path[i+1])] for i in range(len(path) - 1)]
                original_path = nx.shortest_path(
                    self.base,
                    trip.next_node,
                    trip.destination,
                    weight=lambda u, v, _: np.maximum(0,
                                                      self.get_travel_time(u, v, on_edge[(u, v)]))
                )
                original_path_weights = [self.get_travel_time(original_path[i], original_path[i + 1], on_edge[(original_path[i], original_path[i + 1])]) for i in range(len(original_path) - 1)]

                trip_time_diff = sum(path_weights) - sum(original_path_weights)
                assert trip_time_diff >= 0, f'{original_action}'
                component_time_diff[self.base.nodes[trip.next_node]['component']] += trip_time_diff

                trip.prev_node = trip.next_node
                trip.next_node = path[1]
                trip.time_to_next = round(self.get_travel_time(trip.prev_node, trip.next_node,
                                                         on_edge[(trip.prev_node, trip.next_node)]))
                trip.edge_time = trip.time_to_next

                self.logger.log(1, f'Step {self.time_step}.'
                                   f' Trip {trip.number}({trip.start},{trip.destination}) arrived at node {trip.prev_node}.'
                                   f' Going toward {trip.next_node}.'
                                   f' Original path: {original_path}|{original_path_weights}.'
                                   f' Perturbed path: {path}|{path_weights}.'
                                   f' Will arrive there in {trip.time_to_next} steps.'
                                   f' Calculated Time to next is {self.get_travel_time(trip.prev_node, trip.next_node, on_edge[(trip.prev_node, trip.next_node)] + perturbed[(trip.prev_node, trip.next_node)])}.')

            # if vehicle on the edge, let it progress
            if not trip.time_to_next == 0:
                self.logger.log(1, f'Step {self.time_step}.'
                                   f' Trip {trip.number}({trip.start},{trip.destination}) going toward {trip.next_node}.'
                                   f' Will arrive there in {trip.time_to_next} steps.')
                trip.time_to_next -= 1

            if trip.time_to_next == 1 and trip.next_node == trip.destination:
                arrived_vehicles[self.base.nodes[trip.next_node]['component']] += trip.count

        # Fixing the environment observation, reward, done, and info
        if remaining_trips == 0:
            self.finished = True

        done = self.finished

        info = dict(
            original_reward=self.get_reward(),
            perturbed_edge_travel_times=[self.get_travel_time(u, v, on_edge[(u, v)]) + perturbed[(u, v)] for u, v in self.base.edges]
        )

        vehicles_in_component = np.zeros(self.n_components)
        on_edge = self.get_on_edge_vehicles()
        on_vertex = self.get_on_vertex_vehicles()
        for e in on_edge:
            vehicles_in_component[self.base.edges[e]['component']] += on_edge[e]
        for v in on_vertex:
            vehicles_in_component[self.base.nodes[v]['component']] += on_vertex[v]

        info['will_the_attack_count'] = not sum(on_vertex.values()) == 0
        info['vehicles_making_decision'] = sum(on_vertex.values())
        if self.time_step >= self.config['horizon']:
            info['TimeLimit.truncated'] = True

        feature_vector = self.get_current_observation_edge_vector()

        self.finished_previous_step = currently_finished

        if self.config['rewarding_rule'] == 'travel_time_increased':
            reward = component_time_diff
        elif self.config['rewarding_rule'] == 'step_count':
            reward = vehicles_in_component
        elif self.config['rewarding_rule'] == 'normalized':
            mean = np.load(f'{self.base_path}/reward_mean_fixed.npy')
            var = np.load(f'{self.base_path}/reward_var_fixed.npy')
            reward = (vehicles_in_component - mean) / (var + 1e-8)
        elif self.config['rewarding_rule'] == 'proportional':
            reward = vehicles_in_component / self.max_number_of_vehicles
        elif self.config['rewarding_rule'] == 'arrived':
            reward = -arrived_vehicles
        else:
            raise Exception(f'Rewarding rule {self.config["rewarding_rule"]} not implemented.')

        # reward = reward * self.config['reward_multiplier'] - self.config['norm_penalty_coeff'] * component_action_penalty
        # info['norm_penalty'] = sum(component_action_penalty)
        reward *= self.config['reward_multiplier']

        return feature_vector, reward, done, info

    # This function partitions the graph into n_component components:
    def partition_graph(self, n_components, n_repeat=100):

        assert n_components <= self.base.number_of_nodes(), f'Number of components should be less than the number of nodes in the graph. {n_components} >= {self.base.number_of_nodes()}'

        # centroids = random.sample(self.base.nodes, k=n_components)
        centroids = [i + 1 for i in range(n_components)]
        for i, c in enumerate(centroids):
            self.base.nodes[c]['component'] = i
            # nx.set_node_attributes(self.base, {c: {'component': i}})

        all_pairs_distance = dict(nx.all_pairs_dijkstra(
            self.base,
            weight=lambda u, v, _: np.maximum(0, self.base.edges[(u, v)]['free_flow_time'])
        ))  # A dict with key being the 'source_node' and value is (distance, path)

        self.logger.info(f'Partitioning graph into {n_components} components. Repeat: {n_repeat}')
        for _ in range(n_repeat):  # _ is the K-Means clustering algorithm.

            for node in self.base:  # Assigning each node to its closest centroid
                distance_to_centroids = [all_pairs_distance[node][0][c] for c in centroids]
                closest_centroid = centroids[np.argmin(distance_to_centroids)]
                self.base.nodes[node]['component'] = self.base.nodes[closest_centroid]['component']

            # Finding the centroid of each component
            # The centroid of a component is the node with the minimum maximum distance to other nodes in the same component.
            for comp in range(n_components):
                comp_max_distance = np.inf
                for source in self.base:
                    if self.base.nodes[source]['component'] == comp:
                        max_distance = 0
                        for destination in self.base:
                            if self.base.nodes[destination]['component'] == comp:
                                if (distance := all_pairs_distance[source][0][destination]) > max_distance:
                                    max_distance = distance

                        if max_distance < comp_max_distance:
                            comp_max_distance = max_distance
                            centroids[comp] = source

        for e in self.base.edges:
            self.base.edges[e]['component'] = self.base.nodes[e[0]]['component']

    def get_travel_times_assuming_the_attack(self, action):
        action = np.maximum(action, 0)

        on_edge = self.get_on_edge_vehicles()

        perturbed = dict()
        for i, e in enumerate(self.base.edges):
            perturbed[e] = action[i]

        return [self.get_travel_time(u, v, on_edge[(u, v)]) + perturbed[(u, v)] for u, v in self.base.edges]

    def show_base_graph(self):
        pos = nx.spectral_layout(self.base)

        fig, ax = plt.subplots()

        fig.set_figheight(25)
        fig.set_figwidth(25)
        ax.margins(0.0)
        plt.axis("off")
        node_colors = [DynamicMultiAgentTransportationNetworkEnvironment.get_color_by_component(self.base.nodes[n]['component']) for n in self.base.nodes]
        edge_colors = [DynamicMultiAgentTransportationNetworkEnvironment.get_color_by_component(self.base.edges[n]['component']) for n in self.base.edges]

        nx.draw(self.base, pos=pos, ax=ax, node_color=node_colors, edge_color=edge_colors, with_labels=True)
        plt.show()

    @staticmethod
    def get_color_by_component(component: int):
        colors = [
            "#FF0000",  # Red
            "#FFA500",  # Orange
            "#FFFF00",  # Yellow
            "#008000",  # Green
            "#0000FF",  # Blue
            "#4B0082",  # Indigo
            "#EE82EE",  # Violet
            "#800080",  # Purple
            "#FFC0CB",  # Pink
            "#FF00FF",  # Magenta
            "#00FFFF",  # Cyan
            "#40E0D0",  # Turquoise
            "#00FF00",  # Lime
            "#808000",  # Olive
            "#FFD700",  # Gold
            "#C0C0C0",  # Silver
            "#808080",  # Gray
            "#000000",  # Black
            "#FFFFFF",  # White
            "#800000",  # Maroon
            "#A52A2A",  # Brown
            "#F5F5DC",  # Beige
            "#008080",  # Teal
            "#000080",  # Navy
            "#87CEEB",  # Sky Blue
            "#E6E6FA",  # Lavender
            "#DDA0DD",  # Plum
            "#FF7F50",  # Coral
            "#FA8072",  # Salmon
            "#708090"  # Slate
        ]

        return colors[component % len(colors)]

    def __get_edge_component_mapping(self):
        edge_component_mapping = [[] for _ in range(self.n_components)]
        for i, e in enumerate(self.base.edges):
            edge_component_mapping[self.base.edges[e]['component']].append(i)
        return edge_component_mapping

    def get_component_edge_mapping(self):
        return [
            self.base.edges[e]['component'] for e in self.base.edges
        ]
