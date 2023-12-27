import logging
import sys

import gym
import numpy as np
import util.graphing as graphing

from models import attack_heuristics
from models.attack_heuristics import Zero
from models.dl.noise import ZeroNoise
from models.rl_attackers import FixedBudgetNetworkedWideGreedy
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
            file='GRE-4x4-0.5051-0.1111-20231121131109967053',
            # file='GRE-4x2-0.5051-0.1111-20231121124113244125',
            randomize_factor=0.01
        ),
        horizon=50,
        render_mode=None,
        congestion=True,
        # rewarding_rule='normalized',
        rewarding_rule='proportional',
        reward_multiplier=1.0,
        n_components=4,
    )

    logger.info(f'Config: {config}')

    # logger.info(f'Observation Space: {env.observation_space}')
    # logger.info(f'Action Space: {env.action_space}')

    env = DynamicMultiAgentTransportationNetworkEnvironment(config)

    edge_capacities = np.log([env.base.edges[e]['capacity'] for e in env.base.edges])
    edge_powers = np.log([env.base.edges[e]['power'] for e in env.base.edges])
    edge_free_flow_times = np.log([env.base.edges[e]['free_flow_time'] for e in env.base.edges])
    edge_b = np.log([env.base.edges[e]['b'] for e in env.base.edges])
    trip_size = np.log([t.count for t in env.base_trips])

    print(f'edge_capacities_mean: {np.mean(edge_capacities)}, edge_capacities_std: {np.std(edge_capacities)}')
    print(f'edge_powers_mean: {np.mean(edge_powers)}, edge_powers_std: {np.std(edge_powers)}')
    print(f'edge_free_flow_times_mean: {np.mean(edge_free_flow_times)}, edge_free_flow_times_std: {np.std(edge_free_flow_times)}')
    print(f'edge_b_mean: {np.mean(edge_b)}, edge_b_std: {np.std(edge_b)}')
    print(f'trip_size_mean: {np.mean(trip_size)}, trip_size_std: {np.std(trip_size)}')

    attacker = FixedBudgetNetworkedWideGreedy(
        edge_component_mapping=env.edge_component_mapping,
        budget=30,
        budget_noise=ZeroNoise()
    )

    rewards = []
    obs = env.reset()
    done = False
    truncated = False
    while not done and not truncated:
        constructed_action, action, allocations, budgets = attacker.forward_single(obs, True)
        obs, reward, done, info = env.step(action)
        truncated = info.get('TimeLimit.truncated', False)
        original_reward = info['original_reward']
        rewards.append(original_reward)

    print(sum(rewards))

