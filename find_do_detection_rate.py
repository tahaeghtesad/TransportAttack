import logging
import sys

import numpy as np
from rich.progress import Progress

from models.double_oracle.trainer import Trainer
from strategies.defender_strategies import MixedDefenderStrategy, SBDefenderStrategy, StochasticBudgetDefender, \
    NoDefense
from strategies.attacker_strategies import StrategicGaussianAttack, ZeroAttackStrategy, MixedAttackerStrategy
from transport_env.MultiAgentEnv import DynamicMultiAgentTransportationNetworkEnvironment

if __name__ == '__main__':
    logging.basicConfig(
        format='[%(asctime)s] [%(name)s] [%(threadName)s] [%(levelname)s] - %(message)s',
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    env_config = dict(
        network=dict(
            method='edge_list',
            file='GRE-3x2-0.5051-0.1111-20241025130124181010_default',
            # city='SiouxFalls',
            randomize_factor=0.5,
        ),
        horizon=200,
        render_mode=None,
        congestion=True,
        rewarding_rule='proportional',
        reward_multiplier=1.0,
        n_components=4,
    )

    env = DynamicMultiAgentTransportationNetworkEnvironment(env_config)

    # attacker_strategy = StrategicGaussianAttack(env, 1)
    # attacker_strategy = ZeroAttackStrategy()
    # attacker_strategy = MixedAttackerStrategy(
    #     strategies=[StrategicGaussianAttack(env, i) for i in range(0, 4)],
    #     probabilities=[0.25, 0.25, 0.25, 0.25]
    # )

    trainer = Trainer(env_config, dict(
        do_config=dict(
            testing_epochs=2048,
            iterations=10
        )
    ))
    trainer.payoff_table = np.loadtxt('3x3graph_results/payoff_table.csv', delimiter=',')

    probabilities = trainer.solve_defender()[1:]
    probabilities = np.array(probabilities) / sum(probabilities)
    print(probabilities)
    print(-trainer.get_value() * env.max_number_of_vehicles)

    defender_mixed_strategy = MixedDefenderStrategy(
        policies=[SBDefenderStrategy(f'3x3graph_results/ppo_defender_{i}') for i in range(2, 12)],
        probabilities=probabilities
    )
    no_defense = NoDefense()
    no_attack = ZeroAttackStrategy()

    for i in range(4):
        defense_value = trainer.play(StrategicGaussianAttack(env, i), defender_mixed_strategy) * env.max_number_of_vehicles
        no_defense_value = trainer.play(StrategicGaussianAttack(env, i), no_defense) * env.max_number_of_vehicles
        no_attack_value = trainer.play(no_attack, no_defense) * env.max_number_of_vehicles
        msne_value = -trainer.get_value() * env.max_number_of_vehicles
        print(f'Component {i}: MSNE: {msne_value:.2f} | No Attack {no_attack_value:.2f} -> No Defense {no_defense_value:.2f} -> Defense {defense_value:.2f}'
              f' | No Attack -> Attack = {((no_defense_value / no_attack_value) - 1) * 100:.2f}%'
              f' | Defense -> No Attack = {((defense_value / no_attack_value) - 1) * 100:.2f}%'
              f' | Defense -> MSNE = {((defense_value / msne_value) - 1) * 100:.2f}%')
