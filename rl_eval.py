import logging
import logging
import sys
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from models.attack_heuristics import Zero
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
                randomize_factor=0.001,
            ),
            rewarding_rule='proportional',
            n_components=4,
        )
    )

    env = DynamicMultiAgentTransportationNetworkEnvironment(config['env_config'])
    # env.show_base_graph()

    device = torch.device('cpu')
    logger.info(device)

    models = [
        Zero(env.edge_component_mapping),
        FixedBudgetNetworkedWideGreedy(env.edge_component_mapping, 30, None),
        Attacker(
            'FixedBudgetProportionalAllocationGreedyComponent',
            env.edge_component_mapping,
            FixedBudgeting(30, None),
            ProportionalAllocator(),
            GreedyComponent(env.edge_component_mapping)
        ),
        # torch.load('logs/20230912-170344/weights/Attacker_2392.tar', map_location=device),
        # torch.load('logs/20230911-155405/weights/Attacker_4000.tar', map_location=device),
    ]
    logger.info(models)
    num_episodes = 10

    model_rewards = {}

    for model in models:
        rewards = []

        for episode in tqdm(range(num_episodes), desc=f'Evaluating {model.name}'):

            o = env.reset()
            done = False
            truncated = False
            reward = 0
            episode_reward = 0
            while not done and not truncated:
                constructed_action, action, allocation, budget = model.forward_single(o, deterministic=True)

                no, r, done, i = env.step(constructed_action)
                truncated = i.get("TimeLimit.truncated", False)
                original_reward = i['original_reward']
                o = no
                episode_reward += original_reward
            rewards.append(episode_reward)

        model_rewards[model] = rewards

    create_box_plot(
        np.array([model_rewards[model] for model in models]),
        'Strategy Rewards',
        'Strategy',
        [model.name for model in models],
        'Reward',
        None,
        True,
    )
