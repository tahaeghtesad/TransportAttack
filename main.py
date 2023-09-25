import logging
import sys

import gym
import numpy as np
import util.graphing as graphing

from models import attack_heuristics
from models.attack_heuristics import PostProcessHeuristic
from transport_env.MultiAgentEnv import DynamicMultiAgentTransportationNetworkEnvironment

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
            type='trips_file',
            randomize_factor=0.1,
        ),
        horizon=20,
        norm=1,
        frac=0.5,
        num_sample=50,
        render_mode=None,
        # reward_multiplier=0.00001,
        reward_multiplier=1,
        congestion=True,
        rewarding_rule='step_count',
        repeat=50,
        observation_type='vector',
        n_components=4,
        epsilon=30,
    )

    logger.info(f'Config: {config}')

    # logger.info(f'Observation Space: {env.observation_space}')
    # logger.info(f'Action Space: {env.action_space}')

    strategies = [
        PostProcessHeuristic(
            attack_heuristics.Zero((76,))),
        PostProcessHeuristic(
            attack_heuristics.Random((76,), config['norm'], config['epsilon'], config['frac'], 'continuous')),
        PostProcessHeuristic(
            attack_heuristics.Random((76,), config['norm'], config['epsilon'], config['frac'], 'discrete')),
        PostProcessHeuristic(
            attack_heuristics.GreedyRiderVector(config['epsilon'], config['norm']),
        )
    ]

    step_data = np.zeros((config['repeat'], len(strategies)))
    cumulative_reward_data = np.zeros((config['repeat'], len(strategies)))
    discounted_reward_data = np.zeros((config['repeat'], len(strategies)))

    for r_factor in np.logspace(-3, 0, 10):
        config['trips']['randomize_factor'] = r_factor

        env = gym.wrappers.TimeLimit(DynamicMultiAgentTransportationNetworkEnvironment(config), config['horizon'])

        gamma = 0.97

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
                t = False
                step_count = 0

                while not d and not t:
                    a = strategy.predict(o)
                    o, r, d, i = env.step(a)
                    t = i.get('TimeLimit.truncated', False)
                    r = sum(r)
                    cumulative_reward += r
                    discounted_reward += gamma ** env.time_step * r
                    step_count += 1

                discounted_rewards.append(discounted_reward)
                cumulative_rewards.append(cumulative_reward)
                step_counts.append(step_count)

                discounted_reward_data[trial, s_num] = discounted_reward
                cumulative_reward_data[trial, s_num] = cumulative_reward
                step_data[trial, s_num] = step_count

            # logger.info(f'Discounted Reward: '
            #             f'\u03BC={np.mean(discounted_rewards):.2f},'
            #             f'\u03C3\u00B2={np.var(discounted_rewards):.2f},'
            #             f'min={np.min(discounted_rewards):.2f},'
            #             f'q25={np.quantile(discounted_rewards, .25):.2f},'
            #             f'q50={np.quantile(discounted_rewards, .50):.2f},'
            #             f'q75={np.quantile(discounted_rewards, .75):.2f},'
            #             f'max={np.max(discounted_rewards):.2f}')
            # logger.info(f'Cumulative Reward: '
            #             f'\u03BC={np.mean(cumulative_rewards):.2f},'
            #             f'\u03C3\u00B2={np.var(cumulative_rewards):.2f},'
            #             f'min={np.min(cumulative_rewards):.2f},'
            #             f'q25={np.quantile(cumulative_rewards, .25):.2f},'
            #             f'q50={np.quantile(cumulative_rewards, .50):.2f},'
            #             f'q75={np.quantile(cumulative_rewards, .75):.2f},'
            #             f'max={np.max(cumulative_rewards):.2f}')
            # logger.info(f'Step Count: '
            #             f'\u03BC={np.mean(step_counts):.2f},'
            #             f'\u03C3\u00B2={np.var(step_counts):.2f},'
            #             f'min={np.min(step_counts):.2f},'
            #             f'q25={np.quantile(step_counts, .25):.2f},'
            #             f'q50={np.quantile(step_counts, .50):.2f},'
            #             f'q75={np.quantile(step_counts, .75):.2f},'
            #             f'max={np.max(step_counts):.2f}')

        graphing.create_grouped_box_plot(
            data_1=discounted_reward_data,
            data_2=cumulative_reward_data,
            title=f'Strategy Reward with fixed $B={config["epsilon"]}$, $F={r_factor:.3f}$',
            x_label='Attack Strategies',
            x_ticks=['NoAttack', 'Continuous', 'Discrete', 'Greedy'],
            y_label_1=f'Average Discounted Reward, $\\gamma$={gamma:.2f}',
            y_label_2=f'Average Cumulative Reward',
            save_path=f'./results/real_world_data_with_randomized_factor_{r_factor:.3f}.png',
            show=True,
        )
