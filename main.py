import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

import attack_heuristics
from attack_heuristics import PostProcessHeuristic
from transport_env.NetworkEnv import TransportationNetworkEnvironment
from transport_env.model import Trip

if __name__ == '__main__':

    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    logger = logging.getLogger(__name__)

    config = dict(
        city='SiouxFalls',
        horizon=50,
        epsilon=30,
        norm=5,
        frac=0.5,
        num_sample=20,
        render_mode=None,
        reward_multiplier=1.0,
        congestion=True,
        trips=dict(
            type='demand_file',
            trips=Trip.trips_using_demand_file('Sirui/traffic_data/sf_demand.txt'),
            strategy='random',
            count=10
        ),
        rewarding_rule='vehicle_count',
        repeat=10
        )

    logger.info(f'Config: {config}')

    config.update(dict(trip_generator=Trip.using_demand_file(f'Sirui/traffic_data/sf_demand.txt', 'random', 100)))

    # logger.info(f'Observation Space: {env.observation_space}')
    # logger.info(f'Action Space: {env.action_space}')

    strategies = [
        PostProcessHeuristic(
            attack_heuristics.Zero((76, ))),
        PostProcessHeuristic(
            attack_heuristics.Random((76, ), config['norm'], config['epsilon'], config['frac'], 'continuous')),
        PostProcessHeuristic(
            attack_heuristics.Random((76, ), config['norm'], config['epsilon'], config['frac'], 'discrete')),
        # PostProcessHeuristic(
            # attack_heuristics.MultiRandom(env.action_space, config['num_sample'], 'continuous', config['frac'],
            #                               config['norm'], config['epsilon'])),
        # PostProcessHeuristic(
            # attack_heuristics.MultiRandom(env.action_space, config['num_sample'], 'discrete', config['frac'],
            #                               config['norm'], config['epsilon'])),
        PostProcessHeuristic(
            attack_heuristics.GreedyRiderVector(config['epsilon'], config['norm']),
        )
    ]

    rewarding_rules = ['vehicle_count', 'step_count', 'vehicles_finished']
    rewarding_rule_descriptions = [
        'Number of Vehicles Currently in the Network',
        '1 if there is at least one vehicle in the network, 0 otherwise',
        'Number of Vehicles Finished Their Trips in this step'
    ]

    # for rewarding_rule, desc in zip(rewarding_rules, rewarding_rule_descriptions):

    rewarding_rule = 'vehicle_count'
    config.update(dict(rewarding_rule=rewarding_rule))

    step_data = np.zeros((config['repeat'], len(strategies)))
    cumulative_reward_data = np.zeros((config['repeat'], len(strategies)))
    discounted_reward_data = np.zeros((config['repeat'], len(strategies)))

    env = gym.wrappers.TimeLimit(TransportationNetworkEnvironment(config), config['horizon'])

    gamma = 0.95

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
                o, r, d, t, i = env.step(a)
                cumulative_reward += r
                discounted_reward += gamma ** env.time_step * r
                logger.debug(f'Reward: {r:.2f} - Done {d}')
                logger.debug(f'Observation:\n{o}')
                step_count += 1

                if t:
                    break

            discounted_rewards.append(discounted_reward)
            cumulative_rewards.append(cumulative_reward)
            step_counts.append(step_count)

            # data[trial, s_num] = cumulative_reward
            discounted_reward_data[trial, s_num] = discounted_reward
            cumulative_reward_data[trial, s_num] = cumulative_reward
            step_data[trial, s_num] = step_count

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

    # plt.boxplot(discounted_reward_data, labels=[s.name for s in strategies])
    # plt.xticks(rotation=45)
    # plt.ylabel(f'Average Discounted Reward, $\\gamma$={gamma:.3f}')
    # plt.xlabel('Strategy')
    # # plt.title(desc)
    # plt.subplots_adjust(bottom=.4)
    # plt.grid()
    # # plt.savefig(f'disc_reward_{rewarding_rule}.png')
    # plt.show()
    # plt.clf()

    plt.boxplot(cumulative_reward_data, labels=[s.name for s in strategies])
    plt.xticks(rotation=45)
    plt.ylabel(f'Average Cumulative Reward')
    plt.xlabel('Strategy')
    # plt.title(desc)
    plt.subplots_adjust(bottom=.4)
    plt.grid()
    # plt.savefig(f'cumulative_reward_{rewarding_rule}.png')
    plt.show()
    plt.clf()

    plt.boxplot(step_data, labels=[s.name for s in strategies])
    plt.xticks(rotation=45)
    plt.ylabel(f'Average Step')
    plt.xlabel('Strategy')
    plt.subplots_adjust(bottom=.4)
    plt.grid()
    # plt.savefig('step_count.png')
    plt.show()
    plt.clf()
