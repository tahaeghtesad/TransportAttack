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

    random.seed(3)
    np.random.seed(3)

    logging.basicConfig(
        stream=sys.stdout,
        format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s',
        level=logging.INFO
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
                randomize_factor=0.001,
            ),
            # rewarding_rule='proportional',
            rewarding_rule='step_count',
            n_components=6,
        )
    )

    env = DynamicMultiAgentTransportationNetworkEnvironment(config['env_config'])
    # env.show_base_graph()
    print(env.edge_component_mapping)

    device = torch.device('cpu')
    logger.info(device)

    fixed_budget_ddpg_allocator_greedy_component: Attacker = torch.load('logs/good_20231003161323458738/weights/Attacker_89000.tar')
    fixed_budget_proportional_allocator_ddpg_component: Attacker = torch.load('logs/20231003164033332819/weights/Attacker_130000.tar')

    models = [
        Zero(env.edge_component_mapping),
        FixedBudgetNetworkedWideGreedy(env.edge_component_mapping, 30, ZeroNoise()),
        Attacker(
            'FixedBudgetProportionalAllocationGreedyComponent',
            env.edge_component_mapping,
            FixedBudgeting(30, ZeroNoise()),
            ProportionalAllocator(),
            GreedyComponent(env.edge_component_mapping)
        ),
        fixed_budget_ddpg_allocator_greedy_component,
        fixed_budget_proportional_allocator_ddpg_component,
    ]
    logger.info(models)
    num_episodes = 10

    model_rewards = {}

    for model in models:
        rewards = []

        observation_history = []
        reward_history = []

        for episode in tqdm(range(num_episodes), desc=f'Evaluating {model.name}'):

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

            rewards.append(episode_reward)

        model_rewards[model] = rewards
        # print(np.array(observation_history).mean(axis=(0, 1)))
        # print(np.array(observation_history).var(axis=(0, 1)))
        # np.save('observation_mean.npy', np.array(observation_history).mean(axis=0))
        # np.save('observation_var.npy', np.array(observation_history).var(axis=0))
        # np.save('reward_mean.npy', np.array(reward_history).mean(axis=0))
        # np.save('reward_var.npy', np.array(reward_history).var(axis=0))

    create_box_plot(
        np.array([model_rewards[model] for model in models]),
        'Strategy Rewards',
        'Strategy',
        [model.name for model in models],
        'Reward',
        None,
        True,
    )
