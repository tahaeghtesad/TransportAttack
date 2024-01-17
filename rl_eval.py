import logging
import random
import sys
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from models.agents.heuristics.attackers.allocators import ProportionalAllocator
from models.agents.heuristics.attackers.attackers import Zero
from models.agents.heuristics.attackers.budgeting import FixedBudgeting
from models.agents.heuristics.attackers.component import GreedyComponent
from models.agents.rl_agents.attackers.rl_attackers import FixedBudgetNetworkedWideGreedy, Attacker
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
            congestion=True,
            trips=dict(
                type='trips_file',
                randomize_factor=0.05,
            ),
            rewarding_rule='step_count',
            n_components=4,
        )
    )

    env = DynamicMultiAgentTransportationNetworkEnvironment(config['env_config'])
    print(env.edge_component_mapping)

    device = torch.device('cpu')
    logger.info(device)

    for _budget in [5, 10, 15, 30]:

        models = [
            Zero(env.edge_component_mapping),
            FixedBudgetNetworkedWideGreedy(env.edge_component_mapping, _budget),
            Attacker(
                'FixedBudgetProportionalAllocationGreedyComponent',
                env.edge_component_mapping,
                FixedBudgeting(_budget),
                ProportionalAllocator(),
                GreedyComponent(env.edge_component_mapping)
            ),
            torch.load(f'report_weights/{_budget}_ddpg.tar'),
            torch.load(f'report_weights/{_budget}_low_level.tar'),
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

        create_box_plot(
            np.array([model_rewards[model] for model in models]),
            '',
            'Strategy',
            names,
            'Total Travel Time',
            f'/Users/txe5135/Desktop/budget-{_budget}.tikz',
            False,
        )
