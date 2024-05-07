import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from transport_env.MultiAgentEnv import DynamicMultiAgentTransportationNetworkEnvironment
from util import tntp

env = DynamicMultiAgentTransportationNetworkEnvironment(dict(
    network=dict(
        method='network_file',
        city='SiouxFalls',
        randomize_factor=0.0,
    ),
    horizon=200,
    render_mode=None,
    congestion=True,
    rewarding_rule='proportional',
    reward_multiplier=1.0,
    n_components=4,
))

paths = env.reset()[:, 0]
paths = np.divide(paths, np.linalg.norm(paths, ord=2))
paths = np.multiply(paths, 10)

graph = env.base

pos = {
    node: (data['X'], data['Y'])
    for node, data in graph.nodes(data=True)
}

node_component = {
    node: data['component']
    for node, data in graph.nodes(data=True)
}

# Draw the graph
plt.figure()
# plt.rcParams["figure.figsize"] = (100, 100)
# plt.rcParams["figure.autolayout"] = True
plt.axis('off')
nx.draw_networkx(
    graph,
    pos,
    with_labels=True,
    node_size=100,
    font_size=4,
    node_color=[node_component[node] for node in graph.nodes()],
    cmap='twilight',
    vmin=0,
    vmax=10,
    arrows=True,
    arrowsize=5,
    label='Component',
    # edge_color=[],
    width=paths,
    style='-',
    connectionstyle='Arc3, rad=0.05',
)
plt.savefig('graph_orig.png', transparent=True)
# plt.show()

