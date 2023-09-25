import logging
import os
import sys
from datetime import datetime
import random

import numpy as np
import torch

from models.attack_heuristics import GreedyRiderVector
from models.double_oracle.trainer import Trainer
from transport_env.MultiAgentEnv import DynamicMultiAgentTransportationNetworkEnvironment

if __name__ == '__main__':
    run_id = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    os.makedirs(f'logs/{run_id}/weights')

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    logging.basicConfig(
        stream=sys.stdout,
        format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    logger = logging.getLogger('main')

    device = torch.device('cpu')

    config = dict(
        run_id=run_id,
        env_config=dict(
            network=dict(
                method='network_file',
                city='SiouxFalls',
            ),
            trips=dict(
                type='trips_file',
                randomize_factor=0.002,
                # type='trips_file_demand'
            ),
            horizon=50,
            render_mode=None,
            reward_multiplier=0.00001,
            # reward_multiplier=1.0,
            congestion=True,
            rewarding_rule='step_count',
            observation_type='vector',
            n_components=4,
        ),
        do_config=dict(
            max_iterations=6,
            testing_epochs=10,
        ),
        rl_config=dict(
            epochs=1_000,
            buffer_size=50_000,
            batch_size=128,
            high_level_lr=dict(
                actor=0.0005,
                critic=0.005,
            ),
            low_level_lr=dict(
                actor=0.005,
                critic=0.01,
            ),
            gamma=0.97,
            tau=0.002,
            noise=dict(
                std_deviation=0.1,
                target_scale=0.005,
                anneal=3_000,
            ),
            exploration=dict(
                start=0.8,
                end=0.1,
                decay=2000,
            ),
            updates=6
        ),
        classifier_config=dict(
            lr=0.001,
            collection_epochs=32,
            training_epochs=32,
            threshold=0.99,
            chi=0,
            batch_size=128,
        )
    )

    env = DynamicMultiAgentTransportationNetworkEnvironment(config['env_config'])
    trainer = Trainer(config, env, device)
    trainer.attacker_strategy_sets = [
        GreedyRiderVector(30, 1),
        # GreedyRiderVector(10, 1),
        # GreedyRiderVector(5, 1),
    ]

    first_defender = trainer.train_classifier([1/len(trainer.attacker_strategy_sets)] * len(trainer.attacker_strategy_sets))
    trainer.defender_strategy_sets.append(first_defender)
    trainer.payoff_table = [[
        trainer.play(trainer.attacker_strategy_sets[i], trainer.defender_strategy_sets[0]) for i in range(len(trainer.attacker_strategy_sets))
    ]]

    for do_iteration in range(config['do_config']['max_iterations']):
        logger.info(f'Iteration {do_iteration}')

        defender_msne = trainer.solve_defender()
        attacker = trainer.train_attacker(defender_msne)
        trainer.attacker_strategy_sets.append(attacker)
        attacker_payoffs = trainer.get_attacker_payoff(attacker)
        trainer.append_attacker_payoffs(attacker_payoffs)

        attacker_msne = trainer.solve_attacker()
        defender = trainer.train_classifier(attacker_msne)
        trainer.defender_strategy_sets.append(defender)
        defender_payoffs = trainer.get_defender_payoff(defender)
        trainer.append_defender_payoffs(defender_payoffs)
