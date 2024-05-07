import math
import random
from datetime import datetime

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

'''
    Generate random graph with GRE method.
    The method is described in the paper:
    "A Random Road Network Model and Its Effects on Topological Characteristics of Mobile Delay-Tolerant Networks"
    https://ieeexplore.ieee.org/document/6520840
    
    The method is implemented in the following code:
    
    :param p: Probability to remove horizontal and vertical edges
    :param q: Probability to add diagonal edges
    :param rows: Number of rows
    :param columns: Number of columns
'''


def create_grid_graph_gre(
        rows,
        columns,
        p=0.6057,
        q=0.3162,
        capacity_mean=9.014490914785046, capacity_std=0.6349776867781077,
        free_flow_time_mean=1.3365527819035248, free_flow_time_std=0.4068261990880405,
        b_mean=-1.8971199848858813, b_std=0.0,
        power_mean=1.3862943611198908, power_std=0.0,
):
    G = nx.grid_2d_graph(rows, columns, create_using=nx.DiGraph)
    for edge in list(G.edges):
        if edge[0][0] == 0 or edge[0][0] == rows - 1 or edge[0][1] == 0 or edge[0][1] == columns - 1:
            # Always keep rim edges
            continue
        if edge[0][0] % 2 == 0:
            # Horizontal edge
            if random.random() > p:
                G.remove_edge(edge[0], edge[1])
        else:
            # Vertical edge
            if not G.has_edge((edge[0][0] - 1, edge[0][1]), edge[0]):
                # If no horizontal edge exists, keep the vertical edge
                if random.random() > (1 - p):
                    G.remove_edge(edge[0], edge[1])
            else:
                if random.random() > (1 - p):
                    G.remove_edge(edge[0], edge[1])

    # Generate diagonal edges
    for node in G.nodes:
        neighbors = [(node[0] + 1, node[1] + 1), (node[0] + 1, node[1] - 1),
                     (node[0] - 1, node[1] + 1), (node[0] - 1, node[1] - 1)]
        for neighbor in neighbors:
            if neighbor in G.nodes and random.random() < q:
                G.add_edge(node, neighbor)

    for edge in G.edges:
        G.edges[edge]['capacity'] = math.exp(random.gauss(capacity_mean, capacity_std))
        G.edges[edge]['free_flow_time'] = math.exp(random.gauss(free_flow_time_mean, free_flow_time_std))
        G.edges[edge]['b'] = math.exp(random.gauss(b_mean, b_std))
        G.edges[edge]['power'] = math.exp(random.gauss(power_mean, power_std))

    mapping = {(i, j): i * columns + j + 1 for i, j in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    return G


def get_trips(graph: nx.DiGraph, trip_size_mean=6.106722990776694, trip_size_std=0.9312653933756303):
    trips = []
    for i in graph.nodes:
        for j in graph.nodes:
            if i != j:
                trips.append((i, j, math.exp(random.gauss(trip_size_mean, trip_size_std))))
    return trips


def write_trips(trips, name):
    with open(f'TransportationNetworks/Generated/{name}.trips', 'w') as fd:
        for trip in trips:
            fd.write(f'{trip[0]},{trip[1]},{trip[2]}\n')


def gen(rows, columns, p, q, trip_density='high'):
    ncomponents = 2
    tries = 0
    while ncomponents != 1:
        graph = create_grid_graph_gre(rows, columns, p, q)
        ncomponents = nx.number_strongly_connected_components(graph)
        tries += 1
    name = f'GRE-{rows}x{columns}-{p:.4f}-{q:.4f}-{datetime.now().strftime("%Y%m%d%H%M%S%f")}'
    nx.write_edgelist(graph, f'TransportationNetworks/Generated/{name}_{trip_density}.edgelist')
    if trip_density == 'high':
        trips = get_trips(graph, trip_size_mean=8)
    elif trip_density == 'default':
        trips = get_trips(graph)
    elif trip_density == 'low':
        trips = get_trips(graph, trip_size_mean=4)
    write_trips(trips, f'{name}_{trip_density}')
    nx.draw_kamada_kawai(graph, with_labels=True)
    plt.savefig(f'TransportationNetworks/Generated/{name}.png')
    print(f'tries: {tries}')
    print(name)
    print(f'n_edges: {len(graph.edges)}')
    print(f'n_nodes: {len(graph.nodes)}')


if __name__ == '__main__':

    # rows = 4
    # columns = 4

    # gen(rows, columns, 0.5051, 0.1111, 'high')
    # gen(rows, columns, 0.5051, 0.1111, 'default')
    # gen(rows, columns, 0.5051, 0.1111, 'low')
    gen(3, 3, 0.5051, 0.1111, 'default')
