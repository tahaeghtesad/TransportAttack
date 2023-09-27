from ray import tune
from ray import train
from ray.tune.schedulers import AsyncHyperBandScheduler, HyperBandForBOHB
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.bohb import TuneBOHB

from rl_trainer import train_single


def objective(config):
    for report in train_single(**config):
        train.report(report)


search_space = {
    'env_randomize_factor': 0.001,
    'allocator_actor_lr': tune.uniform(0.00001, 0.003),
    'allocator_critic_lr': tune.uniform(0.00001, 0.003),
    'allocator_gamma': tune.uniform(0.95, 0.999),
    'allocator_lam': tune.uniform(0.0, 1.0),
    'allocator_epsilon': tune.uniform(0.01, 0.3),
    'allocator_entropy_coeff': tune.uniform(0.1, 0.5),
    'allocator_value_coeff': tune.uniform(0.1, 2.0),
    'allocator_n_updates': 10,
    'allocator_policy_grad_clip': None,
    'allocator_batch_size': 128,
    'allocator_max_concentration': tune.uniform(2.0, 100.0),
    'allocator_clip_range_vf': None,
    'greedy_epsilon': tune.uniform(0.0, 1.0),
}

if __name__ == '__main__':
    tuner = tune.Tuner(
        objective,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=-1,
            metric='episode_reward',
            mode='max',
            scheduler=HyperBandForBOHB(
                time_attr='training_iteration',
                max_t=1000,
                stop_last_trials=False,
            ),
            search_alg=ConcurrencyLimiter(
                TuneBOHB(),
                max_concurrent=8,
            )
        )
    )
    analysis = tuner.fit()
    print(analysis)
    print(analysis.get_best_result().config)
