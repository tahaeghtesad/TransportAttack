import logging
import sys

import numpy as np
from tqdm import tqdm

from models.attack_heuristics import Zero
from transport_env.MultiAgentEnv import DynamicMultiAgentTransportationNetworkEnvironment

if __name__ == '__main__':

    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    logger = logging.getLogger(__name__)

    config = dict(
        network=dict(
            # method='network_file',
            # city='SiouxFalls',
            method='edge_list',
            # file='GRE-4x4-0.5051-0.1111-20240105112518456990_high',
            file='GRE-4x4-0.5051-0.1111-20240105112519255865_default',
            # file='GRE-4x4-0.5051-0.1111-20240105112519374509_low',
            randomize_factor=0.5,
        ),
        horizon=400,
        render_mode=None,
        congestion=True,
        # rewarding_rule='normalized',
        rewarding_rule='proportional',
        # rewarding_rule='travel_time_increased',
        # rewarding_rule='mixed',
        reward_multiplier=1.0,
        n_components=4,
    )

    logger.info(f'Config: {config}')

    # logger.info(f'Observation Space: {env.observation_space}')
    # logger.info(f'Action Space: {env.action_space}')

    env = DynamicMultiAgentTransportationNetworkEnvironment(config)

    # attacker = FixedBudgetNetworkedWideGreedy(
    #     edge_component_mapping=env.edge_component_mapping,
    #     budget=30,
    #     budget_noise=ZeroNoise()
    # )

    attacker = Zero(env.edge_component_mapping)

    mean_travel_times = []
    mean_rewards = []

    for episode in tqdm(range(100)):
        travel_times = []
        rewards = []
        obs = env.reset()
        done = False
        truncated = False
        while not done and not truncated:
            constructed_action, action, allocations, budgets = attacker.forward_single(obs, True)
            obs, reward, done, info = env.step(action)
            truncated = info.get('TimeLimit.truncated', False)
            original_reward = info['original_reward']
            travel_times.append(original_reward)
            rewards.append(sum(reward))

        mean_travel_times.append(sum(travel_times))
        mean_rewards.append(sum(rewards))

    print(f'travel time:\t{np.mean(mean_travel_times)}\t\u00B1\t{np.std(mean_travel_times):.2f}')
    print(f'reward:\t{np.mean(mean_rewards):.2f}\t\u00B1\t{np.std(mean_rewards):.2f}')

