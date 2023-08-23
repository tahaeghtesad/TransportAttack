import logging
import random
import sys
import pandas as pd
import numpy as np

from tqdm import tqdm

from attack_heuristics import PostProcessHeuristic, GreedyRiderVector, Zero, Random
from transport_env.DynamicMultiAgentNetworkEnv import DynamicMultiAgentTransportationNetworkEnvironment

if __name__ == '__main__':

    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    logger = logging.getLogger(__name__)

    config = dict(
        network=dict(
            method='network_file',
            city='SiouxFalls',
        ),
        trips=dict(
            type='trips_file'
        ),
        horizon=50,
        epsilon=30,
        norm=1,
        frac=0.5,
        num_sample=50,
        render_mode=None,
        reward_multiplier=1.0,
        congestion=True,
        rewarding_rule='step_count',
        repeat=100,
        observation_type='vector',
        n_components=4,
        norm_penalty_coeff=1.0,
        capacity_divider=10000,
        deterministic=False,
    )

    env = DynamicMultiAgentTransportationNetworkEnvironment(config)

    # env.show_base_graph()

    # greedy = PostProcessHeuristic(
    #     Random(
    #         (env.base.number_of_edges(), ),
    #         1,
    #         30,
    #         0.5,
    #         'discrete'
    #     )
    # )

    greedy = GreedyRiderVector(
        30, 1
    )

    zero = PostProcessHeuristic(
        Zero(
            (env.base.number_of_edges(), )
        )
    )

    samples_file = open('samples.csv', 'w')

    for episode in tqdm(range(config['repeat'])):

        obs = env.reset()
        done = False
        truncated = False
        steps = 0
        norm_penalty_episode = 0

        while not done and not truncated:
            strategy = 1 if random.random() < config['frac'] else 0

            if strategy == 1:
                action = greedy.predict(obs)
            else:
                action = zero.predict(obs)

            obs, reward, done, info = env.step(action)
            truncated = info.get('TimeLimit.truncated', False)
            steps += 1

            edge_travel_times = info['perturbed_edge_travel_times']
            samples_file.write(','.join([str(x) for x in edge_travel_times] + [str(int(strategy == 1))]) + '\n')
