import logging
import sys

import matplotlib.pyplot as plt

from transport_env.NetworkEnv import TransportationNetworkEnvironment
from transport_env.model import Trip
from util.visualize import Timer

import numpy as np

import attack_heuristics
from attack_heuristics import PostProcessHeuristic

if __name__ == '__main__':

    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    logger = logging.getLogger(__name__)

    config = dict(
        city='SiouxFalls',
        horizon=100,
        epsilon=4,
        norm=np.inf,
        frac=0.5,
        num_sample=20,
        render_mode=None,
        reward_multiplier=1.0,
        congestion=True,
        repeat=100
    )

    logger.info(f'Config: {config}')

    config.update(dict(trip_generator=Trip.using_demand_file(f'Sirui/traffic_data/sf_demand.txt', 'random', 100)))
    env = TransportationNetworkEnvironment(config)

    logger.info(f'Observation Space: {env.observation_space}')
    logger.info(f'Action Space: {env.action_space}')

    strategies = [
        PostProcessHeuristic(
            attack_heuristics.Zero(env.action_space)),
        PostProcessHeuristic(
            attack_heuristics.Random(env.action_space, config['norm'], config['epsilon'], config['frac'], 'continuous')),
        PostProcessHeuristic(
            attack_heuristics.Random(env.action_space, config['norm'], config['epsilon'], config['frac'], 'discrete')),
        # PostProcessHeuristic(
            # attack_heuristics.MultiRandom(env.action_space, config['num_sample'], 'continuous', config['frac'],
            #                               config['norm'], config['epsilon'])),
        # PostProcessHeuristic(
            # attack_heuristics.MultiRandom(env.action_space, config['num_sample'], 'discrete', config['frac'],
            #                               config['norm'], config['epsilon'])),
        PostProcessHeuristic(
            attack_heuristics.GreedyRiderVector(env)
        )
    ]

    data = np.zeros((config['repeat'], len(strategies)))

    for s_num, strategy in enumerate(strategies):
        logger.info(f'Running Strategy {strategy.name} for {config["repeat"]} trials...')

        discounted_rewards = []
        cumulative_rewards = []
        step_counts = []

        for trial in range(config['repeat']):
            o = env.reset()

            cumulative_reward = 0
            discounted_reward = 0

            d = False
            step_count = 0

            while not d:
                a = strategy.predict(o)
                o, r, d, i = env.step(a)
                cumulative_reward += r
                discounted_reward += 0.99 ** env.time_step * r
                logger.debug(f'Reward: {r:.2f} - Done {d}')
                logger.debug(f'Observation:\n{o}')
                step_count += 1

            discounted_rewards.append(discounted_reward)
            cumulative_rewards.append(cumulative_reward)
            step_counts.append(step_count)

            # data[trial, s_num] = cumulative_reward
            data[trial, s_num] = step_count

        logger.info(f'Discounted Reward: '
                    f'\u03BC={np.mean(discounted_rewards):.2f},'
                    f'\u03C3\u00B2={np.var(discounted_rewards):.2f},'
                    f'min={np.min(discounted_rewards):.2f},'
                    f'q25={np.quantile(discounted_rewards, .25):.2f},'
                    f'q50={np.quantile(discounted_rewards, .50):.2f},'
                    f'q75={np.quantile(discounted_rewards, .75):.2f},'
                    f'max={np.max(discounted_rewards):.2f}')
        logger.info(f'Cumulative Reward: '
                    f'\u03BC={np.mean(cumulative_rewards):.2f},'
                    f'\u03C3\u00B2={np.var(cumulative_rewards):.2f},'
                    f'min={np.min(cumulative_rewards):.2f},'
                    f'q25={np.quantile(cumulative_rewards, .25):.2f},'
                    f'q50={np.quantile(cumulative_rewards, .50):.2f},'
                    f'q75={np.quantile(cumulative_rewards, .75):.2f},'
                    f'max={np.max(cumulative_rewards):.2f}')
        logger.info(f'Step Count: '
                    f'\u03BC={np.mean(step_counts):.2f},'
                    f'\u03C3\u00B2={np.var(step_counts):.2f},'
                    f'min={np.min(step_counts):.2f},'
                    f'q25={np.quantile(step_counts, .25):.2f},'
                    f'q50={np.quantile(step_counts, .50):.2f},'
                    f'q75={np.quantile(step_counts, .75):.2f},'
                    f'max={np.max(step_counts):.2f}')

    plt.boxplot(data, labels=[s.name for s in strategies])
    plt.xticks(rotation=45)
    plt.ylabel(f'Average Cumulative Reward - {config["repeat"]}')
    plt.xlabel('Strategy')
    plt.subplots_adjust(bottom=.4)
    plt.grid()
    plt.show()