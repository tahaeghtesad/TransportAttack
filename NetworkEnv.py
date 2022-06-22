import gym
import util.tntp
import networkx as nx
import matplotlib.pyplot as plt


class TransportationNetworkEnvironment(gym.Env):

    # Config contents:
    # - "city": the network file to load.

    def __init__(self, config: dict):
        self.config = config
        self.network = nx.from_pandas_edgelist(
            df=util.tntp.load_net(path=f'./TransportationNetworks/{self.config["city"]}/{self.config["city"]}_net.tntp'),
            source="init_node",
            target="term_node",
            edge_attr=True,
            create_using=nx.DiGraph
        )



    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self, mode="human"):
        raise NotImplementedError()
