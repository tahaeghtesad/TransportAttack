import os

import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

from models.double_oracle.trainer import Trainer
from strategies.attacker_strategies import ZeroAttackStrategy, MixedAttackerStrategy, StrategicGaussianAttack, \
    SBAttackerStrategy
from strategies.defender_strategies import NoDefense, BaseDefenderStrategy, MixedDefenderStrategy, SBDefenderStrategy, \
    BayesianDetector
from transport_env.AdvEnv import BasicDefenderEnv
from transport_env.MultiAgentEnv import DynamicMultiAgentTransportationNetworkEnvironment
from util.rl.history_environment import HistoryEnvironment


def collect_normal_samples(env_config, n_episodes=100, n_steps=5, history=1):

    env = HistoryEnvironment(BasicDefenderEnv(env_config, ZeroAttackStrategy()), history)

    # if 'normal_observations.npy' exists, return that
    if os.path.exists('normal_observations.npy'):
        print(f'Loading normal episode trajectories from file')
        return np.load('normal_observations.npy')

    print('Collecting normal episode trajectories')

    normal_observations = np.full(shape=(n_episodes, n_steps, env.n_history, env.observation_space.shape[1]), fill_value=np.nan)
    zero_detector = NoDefense()
    max_episode_len = 0

    for episode in tqdm(range(n_episodes)):
        observation, _ = env.reset()
        normal_observations[episode, 0] = observation
        for step in range(n_steps - 1):
            action = zero_detector.predict(observation, None)
            observation, _, _, _, _ = env.step(action)
            normal_observations[episode, step + 1] = observation
            max_episode_len = max(max_episode_len, step + 1)

    np.save('normal_observations.npy', normal_observations)
    return normal_observations


def extract_normal_distribution(normal_observations):

    if os.path.exists(f'bayesian_mean.npy') and os.path.exists(f'bayesian_cov.npy'):
        print(f'Loading normal mean and covariance from file')
        return np.load(f'bayesian_mean.npy'), np.load(f'bayesian_cov.npy')

    print(f'extracting multivariate normal distribution from {normal_observations.shape[0]} episodes, shape: {normal_observations.shape}')
    normal_observations = normal_observations[:, :, 0, :]
    mean = np.mean(normal_observations, axis=0)
    cov = np.cov(normal_observations.reshape(normal_observations.shape[0], -1), rowvar=False)
    print("Mean shape:", mean.shape)
    print("Covariance matrix shape:", cov.shape)

    np.save(f'bayesian_mean.npy', mean)
    np.save(f'bayesian_cov.npy', cov)

    return mean, cov


def test_bayesian_detector(env_config, bayesian_detector):
    print(f'Testing bayesian detector against MSNE')
    init_env = DynamicMultiAgentTransportationNetworkEnvironment(env_config)
    trainer_config = dict(
        do_config=dict(
            testing_epochs=100,
            iterations=10
        ),
        defender_n_history=5
    )

    gaussian_attack = MixedAttackerStrategy(
        strategies=[ZeroAttackStrategy()] + [StrategicGaussianAttack(init_env, i) for i in range(len(init_env.edge_component_mapping))],
        probabilities=[0.5] + [0.5 / len(init_env.edge_component_mapping)] * len(init_env.edge_component_mapping)
    )

    trainer = Trainer(env_config, trainer_config)
    trainer.payoff_table = np.loadtxt('sfgraph_results/payoff_table.csv', delimiter=',')

    def_probabilities = trainer.solve_defender()[1:]
    def_probabilities = np.array(def_probabilities) / sum(def_probabilities)

    defender_mixed_strategy = MixedDefenderStrategy(
        policies=[SBDefenderStrategy(f'sfgraph_results/ppo_defender_{i}') for i in range(2, 12)],
        probabilities=def_probabilities
    )

    att_probabilities = trainer.solve_attacker()[1:]
    att_probabilities = np.array(att_probabilities) / sum(att_probabilities)

    attacker_mixed_strategy = MixedAttackerStrategy(
        strategies=[SBAttackerStrategy(f'sfgraph_results/ppo_attacker_{i}') for i in range(1, 11)],
        probabilities=att_probabilities
    )

    gaussian_bayesian = trainer.play(gaussian_attack, bayesian_detector) * init_env.max_number_of_vehicles
    gaussian_msne = trainer.play(gaussian_attack, defender_mixed_strategy) * init_env.max_number_of_vehicles
    msne_msne = -trainer.get_value() * init_env.max_number_of_vehicles
    msne_bayesian = trainer.play(attacker_mixed_strategy, bayesian_detector) * init_env.max_number_of_vehicles

    print(f'Attack vs Defense')
    print(f'Gaussian vs Bayesian: {gaussian_bayesian}')
    print(f'Gaussian vs MSNE: {gaussian_msne}')
    print(f'MSNE vs Bayesian: {msne_bayesian}')
    print(f'MSNE vs MSNE: {msne_msne}')


if __name__ == '__main__':
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

    observations = collect_normal_samples(env_config, 10)
    mean, cov = extract_normal_distribution(observations)
    detector = BayesianDetector(mean, cov)
    test_bayesian_detector(env_config, detector)
