from datetime import datetime

import numpy as np

from models.double_oracle.trainer import Trainer
from models.rl_attackers import FixedBudgetNetworkedWideGreedy
from transport_env.MultiAgentEnv import DynamicMultiAgentTransportationNetworkEnvironment

if __name__ == '__main__':
    env = DynamicMultiAgentTransportationNetworkEnvironment(dict(
        network=dict(
            # method='network_file',
            # city='SiouxFalls',
            method='edge_list',
            file='GRE-4x4-0.5051-0.1111-20231121131109967053',
            randomize_factor=0.05,
        ),
        horizon=400,
        render_mode=None,
        congestion=True,
        # rewarding_rule='normalized',
        # rewarding_rule='proportional',
        # rewarding_rule='travel_time_increased',
        rewarding_rule='mixed',
        reward_multiplier=1.0,
        n_components=4,
    ))

    config = dict(
        run_id=f'{datetime.now().strftime("%Y%m%d%H%M%S%f")}',
        rl_config=dict(
            gamma=0.95,
            buffer_size=100_000,
            batch_size=64,
            epochs=3072,
        ),
        attacker_config=dict(
            high_level=dict(
                critic_lr=1e-3,
                actor_lr=1e-3,
                tau=0.001,
                gamma=0.99,
                target_allocation_noise_scale=0.001,
                actor_update_steps=2,
                noise=dict(
                    scale=0.5,
                    target=0.001,
                    decay=50_000
                ),
                epsilon=dict(
                    start=1.0,
                    end=0.05,
                    decay=50_000
                )
            ),
            low_level=dict(
                critic_lr=1e-3,
                actor_lr=1e-3,
                tau=0.001,
                gamma=0.99,
                target_allocation_noise_scale=0.001,
                actor_update_steps=2,
                noise=dict(
                    scale=0.5,
                    target=0.001,
                    decay=50_000
                ),
                epsilon=dict(
                    start=1.0,
                    end=0.05,
                    decay=50_000
                )
            )
        ),
        detector_config=dict(
            gamma=0.99,
            lr=1e-3,
            tau=0.001,
            attacker_present_probability=0.5,
            rho=200.0,
            epsilon=dict(
                start=1.0,
                end=0.05,
                decay=5_000
            )
        ),
        do_config=dict(
            testing_epochs=32,
            iterations=8,
        )
    )

    trainer = Trainer(config, env)

    attacker_0 = FixedBudgetNetworkedWideGreedy(env.edge_component_mapping, 30, 0.005)
    trainer.attacker_strategy_sets.append(attacker_0)
    detector_0 = trainer.train_detector([1.0])
    trainer.defender_strategy_sets.append(detector_0)
    attacker_payoff = trainer.get_attacker_payoff(attacker_0)
    trainer.append_defender_payoffs([p[0] for p in attacker_payoff])

    for do_iteration in range(config['do_config']['iterations']):
        print(f'DO iteration {do_iteration}')
        probabilities = trainer.solve_defender()
        attacker_i = trainer.train_attacker(probabilities)
        trainer.attacker_strategy_sets.append(attacker_i)
        payoff = trainer.get_attacker_payoff(attacker_i)
        trainer.append_attacker_payoffs([p[0] for p in payoff])

        probabilities = trainer.solve_attacker()
        defender_i = trainer.train_detector(probabilities)
        trainer.defender_strategy_sets.append(defender_i)
        payoff = trainer.get_defender_payoff(defender_i)
        trainer.append_defender_payoffs([p[0] for p in payoff])

        print(f'New MSNE payoff: {sum(trainer.solve_attacker() * np.array(trainer.payoff_table))}')
