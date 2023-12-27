import random
from typing import Iterable, List, Tuple

import networkx as nx
from dataclasses import dataclass

from util.tntp import import_od


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
    count: int
    demand: int

    @staticmethod
    def random_trip_creator(count):
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

                trips.append((start, end, 0))
                i += 1
            return Trip.get_trips(trips)

        return create_random_trips_starting_at_t_equals_zero

    @staticmethod
    def deterministic_test_trip_creator(count):
        def creator(network: nx.Graph):
            return Trip.get_trips(
                [(i % network.number_of_nodes() + 1, (10 - i) % network.number_of_nodes() + 1, 0) for i in
                 range(count)])

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

            return Trip.get_trips(sample_trips)

        return creator

    @staticmethod
    def trips_using_demand_file(path):
        with open(path, 'r') as f:
            sample_trips = [line.split() for line in f.readlines()]
            return Trip.get_trips(sample_trips)

    @staticmethod
    def trips_using_od_file(path):
        return Trip.get_trips(import_od(path))

    @staticmethod
    def reset_trips(trips, randomize_factor=0):
        for trip in trips:
            trip.prev_node = None
            trip.next_node = trip.start
            trip.progress = 0
            trip.time_to_next = 0
            trip.edge_time = 0
            trip.count = max(0, int(trip.demand * (1 - randomize_factor + 2 * randomize_factor * random.random())))

    @staticmethod
    def get_trips(srcdest: Iterable[Iterable[str | int]]) -> List['Trip']:
        trips = list()
        for i, (source, dest, demand) in enumerate(srcdest):
            demand = int(float(demand))
            if demand > 0:
                trips.append(
                    Trip(
                        number=i,
                        start=int(source),
                        destination=int(dest),
                        progress=0,
                        prev_node=None,
                        next_node=int(source),
                        time_to_next=0,
                        edge_time=0,
                        demand=demand,
                        count=demand
                    )
                )
        return trips

    @staticmethod
    def using_csv_file(path):
        with open(path, 'r') as f:
            sample_trips = [line.rstrip().split(',') for line in f.readlines()]
            return Trip.get_trips(sample_trips)
