import logging
import logging
import os
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
from util.graphing import create_box_plot, create_bar_plot

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
            congestion=True,
            trips=dict(
                type='trips_file',
                randomize_factor=0.05,
            ),
            # rewarding_rule='proportional',
            rewarding_rule='step_count',
            n_components=4,
        )
    )

    env = DynamicMultiAgentTransportationNetworkEnvironment(config['env_config'])
    # env.show_base_graph()
    print(env.edge_component_mapping)

    device = torch.device('cpu')
    logger.info(device)

    for _budget in [10]:

        models = [
            Zero(env.edge_component_mapping),
            FixedBudgetNetworkedWideGreedy(env.edge_component_mapping, _budget, ZeroNoise()),
            Attacker(
                'FixedBudgetProportionalAllocationGreedyComponent',
                env.edge_component_mapping,
                FixedBudgeting(_budget, ZeroNoise()),
                ProportionalAllocator(),
                GreedyComponent(env.edge_component_mapping)
            ),
            torch.load(f'report_weights/{_budget}_ddpg.tar'),
            # torch.load(f'report_weights/{_budget}_low_level.tar'),
            torch.load(f'report_weights/Attacker_final.tar'),
            torch.load(f'report_weights/{_budget}_high_level.tar'),
            torch.load(f'report_weights/{_budget}_hierarchical.tar'),
        ]

        names = ['Zero', 'Greedy', 'Heuristic Hierarchical', 'DDPG', 'Low Level', 'High Level', 'Hierarchical']

        logger.info(len(models))
        num_episodes = 50

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

        # data = np.array([np.mean(value) for key, value in model_rewards.items()])
        # base = data[0]
        # data = (data[1:] - base) / base * 100

        create_box_plot(
            np.array([model_rewards[model] for model in models]),
            f'SiouxFalls - {_budget} Budget',
            'Strategy',
            # [model.name for model in models],
            names,
            'Total Travel Time',
            f'/Users/txe5135/Desktop/budget-{_budget}.png',
            False,
        )
