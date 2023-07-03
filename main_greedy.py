import logging
import sys
import copy

import matplotlib.pyplot as plt
import numpy as np
import gym

import attack_heuristics
from attack_heuristics import PostProcessHeuristic
from transport_env.NetworkEnv import TransportationNetworkEnvironment
from transport_env.model import Trip

if __name__ == '__main__':

    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s',
                        level=logging.INFO
                        )

    logger = logging.getLogger(__name__)

    config = dict(
        network=dict(
            method='network_file',
            city='SiouxFalls',
        ),
        # network=dict(
        #     method='generate',
        #     type='grid',
        #     width=5,
        #     height=3,
        # ),
        # network=dict(
        #     method='generate',
        #     type='line',
        #     num_nodes=10,
        # ),
        # network=dict(
        #     method='generate',
        #     type='cycle',
        #     num_nodes=20,
        # ),
        # city='SiouxFalls',
        horizon=50,
        epsilon=30,
        norm=1,
        frac=0.5,
        num_sample=50,
        render_mode=None,
        reward_multiplier=1.0,
        congestion=True,
        trips=dict(
            type='demand_file',
            demand_file='Sirui/traffic_data/sf_demand.txt',
            strategy='top',
            count=10
        ),
        # trips=dict(
        #     type='deterministic',
        #     count=10,
        # ),
        rewarding_rule='travel_time_increased',
        # rewarding_rule='step_count',
        repeat=100,
        observation_type='vector',
        norm_penalty_coeff=0.01
    )

    logger.info(f'Config: {config}')

    # logger.info(f'Observation Space: {env.observation_space}')
    # logger.info(f'Action Space: {env.action_space}')

    env = TransportationNetworkEnvironment(config)

    print(f'Total number of edges: {env.base.number_of_edges()}')

    # env.show_base_graph()

    heuristic = PostProcessHeuristic(
        attack_heuristics.GreedyRiderVector(config['epsilon'], config['norm']),
    )
    # heuristic = PostProcessHeuristic(
    #     attack_heuristics.Zero((env.base.number_of_edges(),))
    # )
    # heuristic = PostProcessHeuristic(
    #     attack_heuristics.Random((76, ), config['norm'], config['epsilon'], config['frac'], 'discrete')
    # )


    gamma = 0.97

    average_cumulative = []
    average_discounted = []
    average_time_step = []
    average_original_reward = []


    for _ in range(1):

        cumulative_rewards = 0
        discounted_rewards = 0
        original_rewards = 0

        d = False
        t = False
        step_count = 0
        o = env.reset()

        while not d and not t:
            a = heuristic.predict(o)
            o, r, d, i = env.step(a)
            t = i.get('TimeLimit.truncated', False)
            original_rewards += i.get('original_reward')
            cumulative_rewards += r
            discounted_rewards += r * gamma ** step_count
            step_count += 1

        average_discounted.append(discounted_rewards)
        average_cumulative.append(cumulative_rewards)
        average_time_step.append(step_count)
        average_original_reward.append(original_rewards)

    print(f'Average Discounted Reward: {np.mean(average_discounted)}')
    print(f'Average Cumulative Reward: {np.mean(average_cumulative)}')
    print(f'Average Time Step: {np.mean(average_time_step)}')
    print(f'Average Original Reward: {np.mean(average_original_reward)}')

    # plt.plot(discounted_rewards, label='Discounted Reward')
    # plt.plot(cumulative_rewards, label='Reward')
    # plt.fill_between(np.arange(len(cumulative_rewards)), discounted_rewards, alpha=0.5)
    # # plt.plot(np.linspace(0, len(cumulative_rewards), 100), sum(cumulative_rewards) * np.ones(100), label='Sum')
    # plt.legend()
    # plt.grid()
    # plt.show()
    # plt.clf()
    #
    # print(sum(cumulative_rewards))
    # print(sum(discounted_rewards))
