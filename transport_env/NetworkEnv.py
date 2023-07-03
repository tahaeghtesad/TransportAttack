import logging
from collections import deque

import gym
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from transport_env.BaseNetworkEnv import BaseTransportationNetworkEnv


class TransportationNetworkEnvironment(BaseTransportationNetworkEnv):

    # Config contents:
    # - "city": the network file to load.
    # - "trip_generator": a lambda function network -> dict(start, end, time)

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

        self.render_mode = config['render_mode']

        if self.render_mode is not None:
            self.pos = nx.kamada_kawai_layout(self.base)  # positions for all nodes

        self.action_space = gym.spaces.Box(0, self.config['epsilon'], (self.base.number_of_edges(),))

        if self.config['observation_type'] == 'vector':
            self.observation_space = gym.spaces.Box(0, np.inf, (self.base.number_of_edges(), 5,))
        elif self.config['observation_type'] == 'graph':
            self.observation_space = gym.spaces.Graph(
                node_space=gym.spaces.Box(0, np.inf, (0,)),
                edge_space=gym.spaces.Box(0, np.inf, (5,)),
            )
        else:
            raise Exception(f'Unknown observation type: {self.config["observation_type"]}')

        # self.pbar = tqdm(total=self.config['horizon'], desc='Episode')

        self.norm_sum = deque(maxlen=self.config['horizon'])

    def reset(
            self,
    ):

        observation_vector = super().reset()

        if self.config['observation_type'] == 'vector':
            return self.get_current_observation_edge_vector()
        elif self.config['observation_type'] == 'graph':
            return observation_vector

    def step(self, action):

        if not self.initialized:
            raise Exception('Call env.reset() to initialize the network and trips before env.step().')
        if self.finished:
            raise Exception('Previous epoch has ended. Call env.reset() ro reinitialize the network.')

        # self.pbar.set_description(f'Action min {np.min(action):.4f}, max {np.max(action):.4f}, median {np.median(action):.4f}, mean {np.mean(action):.4f}, std {np.std(action):.4f}, norm {np.linalg.norm(action, ord=self.config["norm"]):.2f}')

        action = np.maximum(action, 0)
        action_norm = np.linalg.norm(action, ord=self.config['norm'])
        if action_norm > self.config['epsilon']:
            action = np.divide(action, action_norm, where=action_norm != 0) * self.config['epsilon']

        # self.pbar.update(1)

        self.previous_perturbations = action
        self.time_step += 1

        remaining_trips = 0
        on_edge = self.get_on_edge_vehicles()
        on_vertex = self.get_on_vertex_vehicles()


        # listed_on_edge = [(on_edge[e] if e in on_edge else 0) for e in self.current.edges]
        perturbed = dict()
        for i, e in enumerate(self.base.edges):
            perturbed[e] = action[i]

        currently_finished = 0
        time_diff = 0
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
                trip.prev_node = trip.next_node
                trip.next_node = path[1]
                # This is actually not the weight, but perturbed weight
                trip.time_to_next = self.get_travel_time(trip.prev_node, trip.next_node,
                                                         on_edge[(trip.prev_node, trip.next_node)])
                # trip.time_to_next = self.get_travel_time(trip.prev_node, trip.next_node, 0)
                trip.edge_time = trip.time_to_next

                trip_time_diff = sum(path_weights) - sum(original_path_weights)
                assert trip_time_diff >= 0, f'{action}'
                time_diff += trip_time_diff

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

        if remaining_trips == 0:
            self.finished = True

        done = self.finished
        info = dict(
            original_reward=self.get_reward(),
        )
        if self.time_step >= self.config['horizon']:
            info['TimeLimit.truncated'] = True

        if self.config['observation_type'] == 'vector':
            obs = self.get_current_observation_edge_vector()
        elif self.config['observation_type'] == 'graph':
            obs = self.__get_current_observation_graph()
        else:
            raise Exception(f'Unknown observation type: {self.config["observation_type"]}')

        if self.config['rewarding_rule'] == 'vehicle_count':
            reward = self.get_reward()
        elif self.config['rewarding_rule'] == 'step_count':
            reward = 0 if remaining_trips == 0 else 1
        elif self.config['rewarding_rule'] == 'vehicles_finished':
            reward = self.finished_previous_step - currently_finished
        elif self.config['rewarding_rule'] == 'travel_time_increased':
            reward = time_diff
        else:
            raise Exception(f'Unknown rewarding rule: {self.config["rewarding_rule"]}')

        self.finished_previous_step = currently_finished
        norm_penalty = max(0, action_norm - self.config['epsilon']) ** 2
        reward -= self.config['norm_penalty_coeff'] * norm_penalty
        self.norm_sum.append(norm_penalty)
        info['norm_penalty'] = norm_penalty

        return obs, reward, done, info  # False is truncated.

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

    def render(self, mode='human'):

        if self.render_mode is None:
            return

        fig, ax = plt.subplots()

        fig.set_figheight(25)
        fig.set_figwidth(25)
        ax.margins(0.0)
        plt.axis("off")

        # Drawing vehicles on road
        on_edge = self.get_on_edge_vehicles()
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
                          f'{self.get_travel_time(edge[0], edge[1], on_edge[edge])}/'
                          f'{self.get_travel_time(edge[0], edge[1], on_edge[edge] + self.previous_perturbations[i])}'
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

