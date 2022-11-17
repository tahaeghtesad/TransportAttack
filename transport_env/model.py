import random
import networkx as nx
from dataclasses import dataclass


@dataclass
class Trip:
    number: int
    start: int
    destination: int
    progress: int
    prev_node: int
    next_node: int
    time_to_next: int
    edge_time: int

    @staticmethod
    def trip_creator(count):
        def create_random_trips_starting_at_t_equals_zero(network: nx.Graph):
            trips = []
            i = 0
            while i < count:
                start = random.sample(network.nodes, 1)[0]
                end = random.sample(network.nodes, 1)[0]

                if end == start:
                    continue

                try:  # Checking if there is a path between start and end
                    nx.shortest_path(network, start, end)
                except Exception as e:  # TODO make this more specific
                    continue

                trips.append(
                    Trip(
                        number=i,
                        start=start,
                        destination=end,
                        progress=0,
                        prev_node=None,
                        next_node=start,
                        time_to_next=0,
                        edge_time=0
                    )
                )
            return trips
        return create_random_trips_starting_at_t_equals_zero

    @staticmethod
    def deterministic_test_trip_creator(count):
        def creator(network: nx.Graph):
            trips = []
            for i in range(count):
                start = i % network.number_of_nodes() + 1
                end = (10 - i) % network.number_of_nodes() + 1
                trips.append(
                    Trip(
                        number=i,
                        start=start,
                        destination=end,
                        progress=0,
                        prev_node=None,
                        next_node=start,
                        time_to_next=0,
                        edge_time=0
                    )
                )
            return trips
        return creator

    @staticmethod
    def using_demand_file(path, strategy, count):
        def creator(network: nx.Graph):
            trips = []
            with open(path, 'r') as f:
                sample_trips = [line.split() for line in f.readlines()]

            if strategy == 'top':
                sample_trips = sample_trips[:count]
            elif strategy == 'random':
                sample_trips = random.sample(sample_trips, count)
            elif strategy == 'random-top':
                sample_trips = sample_trips[:random.randint(1, count)]
            else:
                raise Exception('Unknown strategy')

            for i, (source, dest, demand) in enumerate(sample_trips):
                trips.append(
                    Trip(
                        number=i,
                        start=int(source),
                        destination=int(dest),
                        progress=0,
                        prev_node=None,
                        next_node=int(source),
                        time_to_next=0,
                        edge_time=0
                    )
                )
            return trips
        return creator

