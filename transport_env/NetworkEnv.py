import logging
import math
from typing import Optional, Union, Tuple, Callable, List, Dict

import gym
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gym.core import ObsType, RenderFrame

import util.tntp
from transport_env.model import Trip


class TransportationNetworkEnvironment(gym.Env[np.ndarray, np.ndarray]):

    # Config contents:
    # - "city": the network file to load.
    # - "trip_generator": a lambda function network -> dict(start, end, time)

    def __init__(self, config: dict):
        self.logger = logging.getLogger(__name__)

        self.config = config
        must_have_keys = [
            ('city', str),
            ('trip_generator', Callable[[nx.Graph], List[Trip]]),
            ('render_mode', str),
            ('horizon', int)
        ]
        self.render_mode = config['render_mode']

        for key, t in must_have_keys:
            if key not in self.config:
                raise Exception(f'Key "{key}" not specified in config')
            # TODO make this work?
            # Apparently <class 'function'> is not a <class 'Callable'>
            # if type(self.config[key]) is not t:
            #     raise Exception(f'Key "{key}" should be of type "{t}", but got "{type(self.config[key])}"')

        self.base: nx.DiGraph = nx.from_pandas_edgelist(
            df=util.tntp.load_net(
                path=f'./TransportationNetworks/{self.config["city"]}/{self.config["city"]}_net.tntp'),
            source="init_node",
            target="term_node",
            edge_attr=True,
            create_using=nx.DiGraph
        )

        # scc = nx.number_strongly_connected_components(self.base)
        # if scc != 1:
        #     raise Exception(f'The graph of {config["city"]} is not Strongly Connected. SCC count: {scc}')

        self.trips: List[Trip] = []
        self.previous_perturbations: List[int] = []
        self.time_step = 0

        self.initialized = False
        self.finished = False
        if self.render_mode is not None:
            self.pos = nx.kamada_kawai_layout(self.base)  # positions for all nodes

        self.action_space = gym.spaces.Box(-np.inf, np.inf, (self.base.number_of_edges(),))
        self.observation_space = gym.spaces.Box(0, np.inf, shape=(self.base.number_of_nodes(), self.base.number_of_nodes()))

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        super().reset(seed=seed, options=options)

        self.trips: List[Trip] = self.config['trip_generator'](self.base)
        self.previous_perturbations = [0 for _ in range(self.base.number_of_edges())]

        self.initialized = True
        self.finished = False
        self.time_step = 0

        return self.__get_current_observation_matrix()

    def step(self, action) -> Union[
        Tuple[ObsType, float, bool, bool, dict], Tuple[ObsType, float, bool, dict]
    ]:
        if not self.initialized:
            raise Exception('Call env.reset() to initialize the network and trips before env.step().')
        if self.finished:
            raise Exception('Previous epoch has ended. Call env.reset() ro reinitialize the network.')

        self.previous_perturbations = action
        self.time_step += 1

        remaining_trips = 0
        on_edge = self.__get_on_edge_vehicles()
        on_vertex = self.__get_on_vertex_vehicles()
        # listed_on_edge = [(on_edge[e] if e in on_edge else 0) for e in self.current.edges]
        perturbed = dict()
        for i, e in enumerate(self.base.edges):
            perturbed[e] = action[i]

        for trip in self.trips:

            if trip.next_node == trip.destination and trip.time_to_next == 0:  # trip is finished
                # self.logger.debug(f'Step: {self.time_step}. Trip {trip.number} finished.')
                continue

            # trip progressed for 1 step
            remaining_trips += 1
            trip.progress += 1

            # vehicle is on a node, we will calculate the next node and send the vehicle on that destination
            if trip.time_to_next == 0:
                path = nx.shortest_path(self.base,
                                        trip.next_node,
                                        trip.destination,
                                        weight=lambda u, v, info: self.__get_travel_time(u, v,
                                                                                         on_edge[(u, v)]) + perturbed[
                                                                      (u, v)])
                trip.prev_node = trip.next_node
                trip.next_node = path[1]
                # This is actually not the weight, but perturbed weight
                trip.time_to_next = self.__get_travel_time(trip.prev_node, trip.next_node,
                                                           on_edge[(trip.prev_node, trip.next_node)])

                self.logger.debug(f'Step {self.time_step}.'
                                  f' Trip {trip.number}({trip.start},{trip.destination}) arrived at node {trip.prev_node}.'
                                  f' Going toward {trip.next_node}.'
                                  f' Will arrive there in {trip.time_to_next} steps.'
                                  f' Calculated Time to next is {self.__get_travel_time(trip.prev_node, trip.next_node, on_edge[(trip.prev_node, trip.next_node)] + perturbed[(trip.prev_node, trip.next_node)])}.')

            # if vehicle on the edge, let it progress
            if not trip.time_to_next == 0:
                self.logger.debug(f'Step {self.time_step}.'
                                  f' Trip {trip.number}({trip.start},{trip.destination}) going toward {trip.next_node}.'
                                  f' Will arrive there in {trip.time_to_next} steps.')
                trip.time_to_next -= 1

        if remaining_trips == 0 or self.time_step >= self.config['horizon']:
            self.finished = True

        reward = (sum(on_edge.values()) + sum(on_vertex.values())) * self.config['reward_multiplier']
        done = self.finished
        obs = self.__get_current_observation_matrix()
        info = dict()

        return obs, reward, done, info

    """Compute the render frames as specified by render_mode attribute during initialization of the environment.

            The set of supported modes varies per environment. (And some
            third-party environments may not support rendering at all.)
            By convention, if render_mode is:

            - None (default): no render is computed.
            - human: render return None.
              The environment is continuously rendered in the current display or terminal. Usually for human consumption.
            - single_rgb_array: return a single frame representing the current state of the environment.
              A frame is a numpy.ndarray with shape (x, y, 3) representing RGB values for an x-by-y pixel image.
            - rgb_array: return a list of frames representing the states of the environment since the last reset.
              Each frame is a numpy.ndarray with shape (x, y, 3), as with single_rgb_array.
            - ansi: Return a list of strings (str) or StringIO.StringIO containing a
              terminal-style text representation for each time step.
              The text can include newlines and ANSI escape sequences (e.g. for colors).
    """

    def render(self, mode='human') -> Optional[Union[RenderFrame, List[RenderFrame]]]:

        if self.render_mode is None:
            return

        fig, ax = plt.subplots()

        fig.set_figheight(25)
        fig.set_figwidth(25)
        ax.margins(0.0)
        plt.axis("off")

        # Drawing vehicles on road
        on_edge = self.__get_on_edge_vehicles()
        max_ne = max(on_edge.values())
        if max_ne == 0:
            max_ne = 1

        for i, edge in enumerate(self.base.edges):
            if edge in on_edge:
                nx.draw_networkx_edges(
                    self.base,
                    self.pos,
                    edgelist=[edge],
                    width=5 * on_edge[edge] / max_ne,
                    alpha=1.0,
                    edge_color="tab:blue",
                    ax=ax,
                    connectionstyle='arc3, rad=0.1',
                    arrows=True,
                    arrowsize=40 * on_edge[edge] / max_ne
                )

        # Drawing perturbations
        max_p = max(self.previous_perturbations)
        if max_p == 0:
            max_p = 1
        for i, edge in enumerate(self.base.edges):
            nx.draw_networkx_edges(
                self.base,
                self.pos,
                edgelist=[edge],
                width=10 * self.previous_perturbations[i] / max_p,
                alpha=0.2,
                edge_color="tab:red",
                ax=ax,
                connectionstyle='arc3, rad=0.05',
                arrows=True,
                arrowsize=15
            )

        for i, edge in enumerate(self.base.edges):
            nx.draw_networkx_edge_labels(
                self.base,
                self.pos,
                edge_labels={
                    edge: f'{self.base.edges[edge]["capacity"]:.0f}/'
                          f'{on_edge[edge]}/'
                          f'{self.previous_perturbations[i]}/'
                          f'{self.__get_travel_time(edge[0], edge[1], on_edge[edge])}/'
                          f'{self.__get_travel_time(edge[0], edge[1], on_edge[edge] + self.previous_perturbations[i])}'
                },
                font_size=16,
                verticalalignment='baseline',
                horizontalalignment='left',
                bbox=dict(alpha=0)
            )

        # nodes
        options = {"edgecolors": "tab:gray", "node_size": 500, "alpha": 1.0}
        nx.draw_networkx_nodes(self.base, self.pos, node_color='tab:blue', ax=ax, **options)

        if self.render_mode == 'human':
            plt.show()

        if self.render_mode == 'single_rgb_array':
            fig.canvas.draw()

            # Now we can save it to a numpy array.
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            return frame

        if self.render_mode == 'write_to_file':
            plt.savefig(f'TestPlots/{self.time_step}.png')

    # Number of vehicles on edge.
    def __get_on_edge_vehicles(self) -> Dict[Tuple[int, int], int]:
        on_edge = dict()

        for t in self.trips:
            if t.time_to_next > 0:
                if (edge := (t.prev_node, t.next_node)) in on_edge:
                    on_edge[edge] += 1
                else:
                    on_edge[edge] = 1

        for e in self.base.edges:
            if e not in on_edge:
                on_edge[e] = 0

        return on_edge

    # Number of vehicles on nodes
    def __get_on_vertex_vehicles(self) -> Dict[int, int]:
        on_vertex = dict()
        for t in self.trips:
            if t.time_to_next == 0 and t.next_node != t.destination:
                if t.next_node in on_vertex:
                    on_vertex[t.next_node] += 1
                else:
                    on_vertex[t.next_node] = 1

        for n in self.base.nodes:
            if n not in on_vertex:
                on_vertex[n] = 0

        return on_vertex

    def __get_travel_time(self, i: int, j: int, on_edge: int) -> int:
        capacity = self.base[i][j]['capacity']
        free_flow_time = self.base[i][j]['free_flow_time']

        return math.ceil(free_flow_time * (1.0 + 0.15 * (on_edge / capacity) ** 4))

    def __get_current_observation(self) -> Tuple[gym.spaces.GraphInstance, gym.spaces.GraphInstance]:
        # I am going to do something that is not intuitive at all. I return two graphs:
        # (NetworkGraph) one is the original base graph.
        # (DecisionGraph) The second is a graph with:
        # E = {(u,v) | a vehicle is at node 'u' in graph 1 and its destination is 'v')
        on_edge = self.__get_on_edge_vehicles()
        on_vertex = self.__get_on_vertex_vehicles()

        decision_edges = []
        for t in self.trips:
            if t.time_to_next == 0 and t.next_node != t.destination:
                decision_edges.append((t.next_node, t.destination))

        return gym.spaces.GraphInstance(
            nodes=np.array([[0] for i in self.base.nodes]),
            edges=np.array([[self.base.edges[i]['capacity'], self.__get_travel_time(i[0], i[1], on_edge[i]), self.__get_travel_time(i[0], i[1], on_edge[i])] for i in self.base.edges]),
            edge_links=np.array([[i, j] for i, j in self.base.edges])
        ), gym.spaces.GraphInstance(
            nodes=np.array([[on_vertex[i]] for i in self.base.nodes]),
            edges=None,
            edge_links=np.array(decision_edges)
        )

    def __get_current_observation_matrix(self):
        obs = np.zeros((self.base.number_of_nodes(), self.base.number_of_nodes()))
        node_to_index = {n: i for i, n in enumerate(self.base.nodes)}

        on_edge = self.__get_on_edge_vehicles()

        for t in self.trips:
            if t.time_to_next == 0 and t.next_node != t.destination:
                path = nx.shortest_path(self.base, t.next_node, t.destination, weight=lambda u, v, d: on_edge[(u, v)])
                for i in range(len(path) - 1):
                    obs[node_to_index[path[i]], node_to_index[path[i + 1]]] += 1

        return obs
