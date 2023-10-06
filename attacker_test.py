import logging
import logging
import random
import sys
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from models.attack_heuristics import Zero
from models.dl.noise import ZeroNoise
from models.heuristics.allocators import ProportionalAllocator
from models.heuristics.budgeting import FixedBudgeting
from models.heuristics.component import GreedyComponent
from models.rl_attackers import FixedBudgetNetworkedWideGreedy, Attacker
from transport_env.MultiAgentEnv import DynamicMultiAgentTransportationNetworkEnvironment
from util.graphing import create_box_plot

if __name__ == '__main__':
    run_id = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    logging.basicConfig(
        stream=sys.stdout,
        format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s',
        level=logging.CRITICAL
    )

    logger = logging.getLogger('main')

    config = dict(
        env_config=dict(
            network=dict(
                method='network_file',
                city='SiouxFalls',
            ),
            horizon=50,
            num_sample=20,
            render_mode=None,
            reward_multiplier=1.0,
            congestion=False,
            trips=dict(
                type='trips_file',
                randomize_factor=0.000,
            ),
            # rewarding_rule='proportional',
            rewarding_rule='step_count',
            n_components=4,
        )
    )

    model_rewards = {}
    reference = 0

    for run in range(100):

        env = DynamicMultiAgentTransportationNetworkEnvironment(config['env_config'])

        metric_name = f'{"|".join([f"{v:0.3f}" for v in env.metrics.values()])}'

        model = Attacker(
            name=metric_name,
            edge_component_mapping=env.edge_component_mapping,
            component=GreedyComponent(env.edge_component_mapping),
            budgeting=FixedBudgeting(30, ZeroNoise()),
            allocator=ProportionalAllocator(),
        ) if run != 0 else FixedBudgetNetworkedWideGreedy(
            env.edge_component_mapping,
            30,
            ZeroNoise(),
        )

        num_episodes = 1

        rewards = []

        observation_history = []
        reward_history = []

        for episode in tqdm(range(num_episodes), desc=f'Evaluating {run}', disable=True):

            o = env.reset()
            done = False
            truncated = False
            reward = 0
            episode_reward = 0
            while not done and not truncated:
                constructed_action, action, allocation, budget = model.forward_single(o, deterministic=False)

                observation_history.append(o)

                no, r, done, i = env.step(constructed_action)

                reward_history.append(r)

                truncated = i.get("TimeLimit.truncated", False)
                original_reward = i['original_reward']
                o = no
                episode_reward += original_reward

            rewards = episode_reward

            if run == 0:
                reference = episode_reward

        model_rewards[model] = rewards
        if model_rewards[model] > reference:
            print(f'{model.name:50}: \033[92m {model_rewards[model]}\033[00m')
        else:
            print(f'{model.name:50}: {model_rewards[model]}')
        # print(np.array(observation_history).mean(axis=(0, 1)))
        # print(np.array(observation_history).var(axis=(0, 1)))
        # np.save('observation_mean.npy', np.array(observation_history).mean(axis=0))
        # np.save('observation_var.npy', np.array(observation_history).var(axis=0))
        # np.save('reward_mean.npy', np.array(reward_history).mean(axis=0))
        # np.save('reward_var.npy', np.array(reward_history).var(axis=0))

    # create_box_plot(
    #     np.array([model_rewards[model] for model in model_rewards.keys()]),
    #     'Strategy Rewards',
    #     'Strategy',
    #     [model.name for model in model_rewards.keys()],
    #     'Reward',
    #     None,
    #     True,
    # )
