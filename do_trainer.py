import logging
import sys
from datetime import datetime

import numpy as np

from models.double_oracle.trainer import Trainer
from strategies.attacker_strategies import ZeroAttackStrategy, SBAttackerStrategy
from strategies.defender_strategies import SBDefenderStrategy, NoDefense
from transport_env.MultiAgentEnv import DynamicMultiAgentTransportationNetworkEnvironment

if __name__ == '__main__':

    logging.basicConfig(
        format='[%(asctime)s] [%(name)s] [%(threadName)s] [%(levelname)s] - %(message)s',
        level=logging.WARNING,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    env_config = dict(
        network=dict(
            # method='edge_list',
            # file='GRE-3x2-0.5051-0.1111-20241025130124181010_default',
            method='network_file',
            city='SiouxFalls',
            randomize_factor=0.5,
        ),
        horizon=50,
        render_mode=None,
        congestion=True,
        rewarding_rule='proportional',
        reward_multiplier=1.0,
        n_components=4,
    )

    env = DynamicMultiAgentTransportationNetworkEnvironment(env_config)

    config = dict(
        run_id=f'{datetime.now().strftime("%Y%m%d%H%M%S%f")}',
        n_envs=128,
        attacker_training_steps=2_000_000,
        defender_training_steps=2_000_000,
        defender_n_history=5,
        do_config=dict(
            iterations=5,
            testing_epochs=64
        )
    )

    trainer = Trainer(env_config, config).to('cuda:1')

    attacker_0 = ZeroAttackStrategy()
    trainer.attacker_strategy_sets.append(attacker_0)

    defender_0 = NoDefense()
    trainer.defender_strategy_sets.append(defender_0)

    trainer.payoff_table = [[trainer.play(attacker_0, defender_0)]]

    logging.getLogger('Base').warning(f'Initial Payoff = {trainer.payoff_table[0][0]}')

    values = []

    for do_iteration in range(config['do_config']['iterations']):

        logging.getLogger('Base').info(f'DO iteration {do_iteration + 1}')

        probabilities = trainer.solve_defender()  # get the mixed strategy of defender
        attacker_i = trainer.train_attacker(probabilities)  # get the best response of attacker
        attacker_i = SBAttackerStrategy(attacker_i)
        trainer.attacker_strategy_sets.append(attacker_i)  # append the attacker strategy to the attacker strategy set
        payoffs = trainer.get_attacker_payoff(attacker_i)  # get the payoff of the attacker
        trainer.append_attacker_payoffs(payoffs)  # append the payoff to the payoff table
        trainer.store_payoff_table()

        probabilities = trainer.solve_attacker()  # get the mixed strategy of attacker
        defender_i = trainer.train_detector(probabilities)  # get the best response of defender
        defender_i = SBDefenderStrategy(defender_i)
        trainer.defender_strategy_sets.append(defender_i)  # append the defender strategy to the defender strategy set
        payoffs = trainer.get_defender_payoff(defender_i)  # get the payoff of the defender
        trainer.append_defender_payoffs(payoffs)
        trainer.store_payoff_table()

        logging.getLogger('Base').warning(f'Attacker Strat = {trainer.solve_attacker()}')
        logging.getLogger('Base').warning(f'Defender Strat = {trainer.solve_defender()}')
        logging.getLogger('Base').warning(f'Payoff Value for Defender = {trainer.get_value()}')
        logging.getLogger('Base').warning(f'Payoff Table =\n{np.array(trainer.payoff_table)}')
        values.append(trainer.get_value())

    logging.getLogger('Base').warning(f'Values = {values}')
