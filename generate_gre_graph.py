import math
import random
from datetime import datetime

import networkx as nx
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
        capacity_mean=10247.206327210528, capacity_std=7310.082537530922,
        free_flow_time_mean=4.131578947368421, free_flow_time_std=1.7194101288336459,
        b_mean=0.15, b_std=0.0,
        power_mean=4.0, power_std=0.0,
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
        G.edges[edge]['capacity'] = max(random.gauss(capacity_mean, capacity_std), 1)
        G.edges[edge]['free_flow_time'] = max(random.gauss(free_flow_time_mean, free_flow_time_std), 1)
        G.edges[edge]['b'] = max(random.gauss(b_mean, b_std), 0)
        G.edges[edge]['power'] = max(random.gauss(power_mean, power_std), 0)

    mapping = {(i, j): i * columns + j + 1 for i, j in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    return G


if __name__ == '__main__':

    rows = 6
    columns = 8

    graph = create_grid_graph_gre(rows, columns)
    while nx.number_strongly_connected_components(graph) != 1:
        graph = create_grid_graph_gre(0, rows, columns)
        print('retry')
    name = f'GRE-{rows}x{columns}-{datetime.now().strftime("%Y%m%d%H%M%S%f")}.edgelist'
    nx.write_edgelist(graph, f'TransportationNetworks/Generated/{name}')
    nx.draw_kamada_kawai(graph, with_labels=True)
    plt.show()
    print(name)
    print(f'n_edges: {len(graph.edges)}')
    print(f'n_nodes: {len(graph.nodes)}')
