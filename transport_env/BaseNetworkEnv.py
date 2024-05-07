import logging
import math
import random
from abc import ABC
from typing import Dict, Tuple, List

import gym
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from transport_env.TripModel import Trip
from util import tntp


class BaseTransportationNetworkEnv(gym.Env, ABC):
    def __init__(self, config, base_path='.'):
        super().__init__()

        self.logger = logging.getLogger(__name__)

        self.config = config
        self.base_path = base_path

        if config['network']['method'] == 'network_file':
            city = config['network']['city']
            self.logger.info(f'Loading network for city {city}')
            self.base: nx.DiGraph = nx.from_pandas_edgelist(
                df=tntp.load_net(
                    path=f'{base_path}/TransportationNetworks/{city}/{city}_net.tntp'),
                source="init_node",
                target="term_node",
                edge_attr=True,
                create_using=nx.DiGraph
            )
            nx.set_node_attributes(
                self.base,
                tntp.load_locations(f'{base_path}/TransportationNetworks/{city}/{city}_node.tntp')
                    .set_index('Node', inplace=False)
                    .to_dict(orient='index')
            )

            self.logger.info(f'City {city} has {self.base.number_of_nodes()} nodes and {self.base.number_of_edges()} edges.')

        elif config['network']['method'] == 'generate':
            self.base = BaseTransportationNetworkEnv.gen_network(**config['network'])
        elif config['network']['method'] == 'edge_list':
            self.base = nx.read_edgelist(
                f'{base_path}/TransportationNetworks/Generated/{config["network"]["file"]}.edgelist',
                create_using=nx.DiGraph,
                data=True,
                nodetype=int,
            )
        else:
            raise Exception(f'Unknown network method: {config["network"]["method"]}'
                            f', available methods: network_file, generate')

        if config['network']['method'] == 'edge_list':
            self.base_trips = Trip.using_csv_file(f'{base_path}/TransportationNetworks/Generated/{config["network"]["file"]}.trips')
        elif self.config['network']['method'] == 'network_file':
            city = self.config['network']['city']
            self.base_trips = Trip.trips_using_od_file(f'{base_path}/TransportationNetworks/{city}/{city}_trips.tntp')
        elif self.config['trips']['type'] == 'trips_file_demand':
            self.base_trips = Trip.using_demand_file(f'{base_path}/traffic_data/sf_demand.txt', 'top', 10)(self.base)
        elif self.config['trips']['type'] == 'deterministic':
            self.base_trips: List[Trip] = Trip.deterministic_test_trip_creator(self.config['trips']['count'])(self.base)
        elif self.config['trips']['type'] == 'random':
            self.base_trips: List[Trip] = Trip.random_trip_creator(self.config['trips']['count'])(self.base)
        else:
            raise Exception(f'Unknown trip type: {self.config["trips"]["type"]}')

        self.logger.info(f'Loaded {len(self.base_trips)}')

        self.time_step = 0

        self.initialized = False
        self.finished = False
        self.finished_previous_step = 0
        self.max_number_of_vehicles = max([t.demand for t in self.base_trips])
        self.total_number_of_vehicles = sum([t.demand for t in self.base_trips])

        self.logger.info(f'Max Number of Vehicles: {self.max_number_of_vehicles}')
        self.logger.info(f'Total Number of Vehicles: {self.total_number_of_vehicles}')

    @property
    def num_edges(self):
        return self.base.number_of_edges()

    @property
    def num_nodes(self):
        return self.base.number_of_nodes()

    def render(self, mode="human"):
        raise NotImplementedError("What are you trying to render?")

    def reset(
            self,
    ):
        self.__reset_trips(randomize_factor=self.config['network']['randomize_factor'])

        self.initialized = True
        self.finished = False
        self.time_step = 0

        self.finished_previous_step = 0

        return self.get_current_observation_edge_vector()

    def __reset_trips(self, randomize_factor=0):
        # self.trips = random.choices(self.base_trips, k=int(random.random() * len(self.base_trips)))
        self.trips = self.base_trips
        Trip.reset_trips(self.trips, randomize_factor)

    # Number of vehicles on edge.
    def get_on_edge_vehicles(self) -> Dict[Tuple[int, int], int]:
        on_edge = dict()

        for t in self.trips:
            if t.time_to_next > 0 and t.prev_node is not None:
                if (edge := (t.prev_node, t.next_node)) in on_edge:
                    on_edge[edge] += t.count
                else:
                    on_edge[edge] = t.count

        for e in self.base.edges:
            if e not in on_edge:
                on_edge[e] = 0

        return on_edge

    # Number of vehicles on nodes
    def get_on_vertex_vehicles(self) -> Dict[int, int]:
        on_vertex = dict()
        for t in self.trips:
            if t.time_to_next == 0 and t.next_node != t.destination:
                if t.next_node in on_vertex:
                    on_vertex[t.next_node] += t.count
                else:
                    on_vertex[t.next_node] = t.count

        for n in self.base.nodes:
            if n not in on_vertex:
                on_vertex[n] = 0

        return on_vertex

    # def get_travel_time(self, i: int, j: int, on_edge: int) -> int:
    #     capacity = self.base[i][j]['capacity']
    #     free_flow_time = self.base[i][j]['free_flow_time']
    #     b = self.base[i][j]['b']
    #     power = self.base[i][j]['power']
    #
    #     return free_flow_time * (1.0 + b * (on_edge * self.config['congestion'] / capacity) ** power)

    def get_travel_time(self, i: int, j: int, d: dict, on_edge: int) -> float:
        return d['free_flow_time'] * (1.0 + d['b'] * (on_edge * self.config['congestion'] / d['capacity']) ** d['power'])

    # Vector of size (E, 5) where E is the number of edges.
    # Each row is a vector of size 5:
    # [
    #   0: # of vehicles on nodes that pass through e as their shortest path without perturbations,
    #   1: Current # of vehicles on e,
    #   2: # of vehicles on nodes that immediately pass e as their shortest path without perturbations,
    #   3: sum of remaining travel time of vehicles on e,
    #   4: # of vehicles that previously had decided on taking edge `e` at some point as their shortest path.
    # ]
    def get_current_observation_edge_vector(self):
        feature_vector = np.zeros((self.base.number_of_edges(), 5,), dtype=np.float32)

        on_edge = self.get_on_edge_vehicles()
        edges = {e: i for i, e in enumerate(self.base.edges)}

        for t in self.trips:
            if t.time_to_next == 0 and t.next_node != t.destination:
                path = nx.shortest_path(
                    self.base, t.next_node, t.destination,
                    weight=lambda u, v, d: self.get_travel_time(u, v, d, on_edge[(u, v)]))
                for i in range(len(path) - 1):
                    feature_vector[edges[(path[i], path[i + 1])]][0] += t.count  # feature 0
                feature_vector[edges[(path[0], path[1])]][2] += t.count  # feature 2
            if t.time_to_next != 0:
                path = nx.shortest_path(
                    self.base, t.next_node, t.destination,
                    weight=lambda u, v, d: self.get_travel_time(u, v, d, on_edge[(u, v)]))
                for i in range(len(path) - 1):
                    feature_vector[edges[(path[i], path[i + 1])]][4] += t.count  # feature 4
                if t.prev_node is not None:
                    edge = (t.prev_node, t.next_node)
                    feature_vector[edges[edge]][3] += (t.time_to_next / self.get_travel_time(*edge, self.base.get_edge_data(*edge), on_edge[edge])) * t.count  # feature 3
                    # feature_vector[edges[edge]][5] += t.count  # feature 5

        for i, e in enumerate(self.base.edges):
            feature_vector[i][1] = on_edge[e]  # feature 1

        return feature_vector / self.max_number_of_vehicles

    def get_adjacency_matrix(self):
        adjacency_matrix = np.zeros((self.base.number_of_edges(), self.base.number_of_edges()), dtype=np.float32)

        for i, e in enumerate(self.base.edges):
            for j, a in enumerate(self.base.edges):
                if e[1] == a[0]:
                    adjacency_matrix[j][i] = 1

        return adjacency_matrix

    def get_adjacency_list(self):
        adjacency_list = list()

        for i, e in enumerate(self.base.edges):
            for j, a in enumerate(self.base.edges):
                if e[1] == a[0]:
                    adjacency_list.append((j, i))

        return adjacency_list

    @staticmethod
    def gen_network(**config):
        network = nx.DiGraph()
        if config['type'] == 'line':
            for i in range(config['num_nodes']):
                network.add_node(i + 1)
                if i != config['num_nodes'] - 1:
                    network.add_edge(i + 1, i + 2, capacity=1000, free_flow_time=i % 3 + 4)
                    network.add_edge(i + 2, i + 1, capacity=1000, free_flow_time=i % 3 + 4)
        elif config['type'] == 'cycle':
            for i in range(config['num_nodes']):
                network.add_node(i + 1)
                if i != config['num_nodes'] - 1:
                    network.add_edge(i + 1, i + 2, capacity=1000, free_flow_time=i % 3 + 4)
                    network.add_edge(i + 2, i + 1, capacity=1000, free_flow_time=i % 3 + 4)
            network.add_edge(1, config['num_nodes'], capacity=1000, free_flow_time=config['num_nodes'] % 3 + 4)
            network.add_edge(config['num_nodes'], 1, capacity=1000, free_flow_time=config['num_nodes'] % 3 + 4)
        elif config['type'] == 'grid':
            width, height = config['width'], config['height']
            for i in range(width):
                for j in range(height):
                    network.add_node(j * width + i + 1)
            for i in range(width):
                for j in range(height):
                    # add right edge
                    if i != width - 1:
                        network.add_edge(j * width + i + 1, j * width + i + 2, capacity=1000, free_flow_time=(i * height + j + 1) % 3 + 4)
                        network.add_edge(j * width + i + 2, j * width + i + 1, capacity=1000, free_flow_time=(i * height + j + 1) % 3 + 4)

                    if j != height - 1:
                        network.add_edge(j * width + i + 1, (j + 1) * width + i + 1, capacity=1000, free_flow_time=(i * height + j + 1) % 3 + 4)
                        network.add_edge((j + 1) * width + i + 1, j * width + i + 1, capacity=1000, free_flow_time=(i * height + j + 1) % 3 + 4)

        else:
            raise Exception(f'Network type "{config["type"]}" not supported'
                            f', supported types are "line" and "grid"')

        return network

    def show_base_graph(self):
        # pos = nx.kamada_kawai_layout(self.base)
        pos = nx.spring_layout(self.base)

        fig, ax = plt.subplots()

        fig.set_figheight(25)
        fig.set_figwidth(25)
        ax.margins(0.0)
        plt.axis("off")

        nx.draw(self.base, pos=pos, ax=ax, with_labels=True)
        plt.show()

    def get_reward(self):
        return sum(self.get_on_vertex_vehicles().values()) + sum(self.get_on_edge_vehicles().values())
