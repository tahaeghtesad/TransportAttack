from datetime import datetime

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
            epochs=512,
        ),
        attacker_config=dict(
            high_level=dict(
                critic_lr=1e-3,
                actor_lr=1e-3,
                tau=0.001,
                gamma=0.99,
                target_allocation_noise=0.001,
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
                target_allocation_noise=0.001,
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
            rho=5.0,
        ),
        do_config=dict(
            testing_epochs=64,
        )
    )

    trainer = Trainer(config, env)

    trainer.attacker_strategy_sets.append(FixedBudgetNetworkedWideGreedy(env.edge_component_mapping, 30, 0.005))
    trainer.train_detector([1.0])