import logging
import sys

import numpy as np
from tqdm import tqdm

from attack_heuristics import PostProcessHeuristic, GreedyRiderVector, Zero
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
        horizon=20,
        norm=1,
        frac=0.5,
        num_sample=50,
        render_mode=None,
        # reward_multiplier=0.00001,
        reward_multiplier=1,
        congestion=True,
        rewarding_rule='step_count',
        repeat=1,
        observation_type='vector',
        n_components=4,
        deterministic=False
    )

    env = DynamicMultiAgentTransportationNetworkEnvironment(config)

    # env.show_base_graph()

    step_counts = []
    reward_sum = []

    # greedy = PostProcessHeuristic(
    #     GreedyRiderVector(
    #         20,
    #         config['norm']
    #     )
    # )

    greedy = Zero(
        (env.base.number_of_edges(), )
    )

    for episode in tqdm(range(config['repeat'])):

        obs = env.reset()
        done = False
        truncated = False
        steps = 0
        norm_penalty_episode = 0
        rewards = 0

        while not done and not truncated:
            # action = [
            #     obs[comp][:, 0] for comp in range(env.n_components)
            # ]
            action = greedy.predict(obs)
            obs, reward, done, info = env.step(action)
            obs = obs
            # logger.info(f'Action: {[a.shape for a in action]}, Obs: {[s.shape for s in obs]}, Reward: {reward}, Done: {done}, Info: {info}')
            truncated = info.get('TimeLimit.truncated', False)
            rewards += sum(reward)
            steps += 1

        reward_sum += [rewards]

    print(f'mean: {np.mean(reward_sum)} | std: {np.std(reward_sum)} | min {np.min(reward_sum)} | max {np.max(reward_sum)}')
