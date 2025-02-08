import json

import numpy as np
import tikzplotlib
from matplotlib import pyplot as plt

from models.double_oracle.trainer import Trainer
from strategies.attacker_strategies import MixedAttackerStrategy, SBAttackerStrategy, StrategicGaussianAttack, \
    ZeroAttackStrategy, GreedyAttackStrategy
from strategies.defender_strategies import MixedDefenderStrategy, SBDefenderStrategy, BayesianDetector, NoDefense
from transport_env.MultiAgentEnv import DynamicMultiAgentTransportationNetworkEnvironment
from util.math import solve_lp


def extract_comparison(config):

    print(f'Loading {config}')

    with open(f'{config}graph_results/env_config.json') as f:
        env_config = json.load(f)

    with open(f'{config}graph_results/trainer_config.json') as f:
        trainer_config = json.load(f)

    init_env = DynamicMultiAgentTransportationNetworkEnvironment(env_config)

    # Loading Heuristics
    print(f'Loading Heuristics for {config}')

    gaussian_attackers = [StrategicGaussianAttack(init_env, i) for i in range(len(init_env.edge_component_mapping))]
    zero_attacker = ZeroAttackStrategy()
    greedy_attacker = GreedyAttackStrategy(budget=30.0)
    bayesian_detector = BayesianDetector.from_normal_observation(f'{config}graph_results/normal_observations.npy')
    zero_defender = NoDefense()

    # Loading MSNE
    print(f'Loading MSNE for {config}')

    trainer = Trainer(env_config, trainer_config)
    trainer.payoff_table = np.loadtxt(f'{config}graph_results/payoff_table.csv', delimiter=',')

    def_probabilities = trainer.solve_defender()[1:]
    def_probabilities = np.array(def_probabilities) / sum(def_probabilities)

    defender_mixed_strategy = MixedDefenderStrategy(
        policies=[SBDefenderStrategy(f'{config}graph_results/ppo_defender_{i}') for i in range(2, 12)],
        probabilities=def_probabilities
    )

    att_probabilities = trainer.solve_attacker()[1:]
    att_probabilities = np.array(att_probabilities) / sum(att_probabilities)

    attacker_mixed_strategy = MixedAttackerStrategy(
        strategies=[SBAttackerStrategy(f'{config}graph_results/ppo_attacker_{i}') for i in range(1, 11)],
        probabilities=att_probabilities
    )

    # Calculating Payoffs
    #attacker - defender

    print(f'Calculating Payoffs for {config}')

    msne_msne = trainer.play_all(attacker_mixed_strategy, defender_mixed_strategy)

    # attackers
    gaussian_0_msne = trainer.play_all(gaussian_attackers[0], defender_mixed_strategy)
    gaussian_1_msne = trainer.play_all(gaussian_attackers[1], defender_mixed_strategy)
    gaussian_2_msne = trainer.play_all(gaussian_attackers[2], defender_mixed_strategy)
    gaussian_3_msne = trainer.play_all(gaussian_attackers[3], defender_mixed_strategy)
    greedy_attacker_msne = trainer.play_all(greedy_attacker, defender_mixed_strategy)

    # defenders
    msne_bayesian_detector = trainer.play_all(attacker_mixed_strategy, bayesian_detector)
    msne_no_defense = trainer.play_all(attacker_mixed_strategy, zero_defender)

    no_attack_no_defense = trainer.play_all(zero_attacker, zero_defender)

    # Let's plot

    print(f'Plotting for {config}')

    data = np.array([no_attack_no_defense, msne_msne, gaussian_0_msne, gaussian_1_msne, gaussian_2_msne, gaussian_3_msne, greedy_attacker_msne, msne_no_defense, msne_bayesian_detector]) * init_env.max_number_of_vehicles
    labels = np.array(['Nominal', 'MSNE', 'Gaussian-1', 'Gaussian-2', 'Gaussian-3', 'Gaussian-4', 'Greedy', 'No Defense', 'Bayesian'])

    attack_indices = [0, 1, 2, 3, 4, 5, 6]
    defense_indices = [0, 1, 7, 8]
    plot(config, data[attack_indices, :], labels[attack_indices], 'attack')
    plot(config, data[defense_indices, :], labels[defense_indices], 'defense')


def solve_attacker(table):
    return solve_lp(-np.array(table))


def solve_defender(table):
    return solve_lp(np.transpose(np.array(table)))


def get_value(table):
    return solve_defender(table) @ -np.array(table) @ solve_attacker(table)


def extract_do_plot(config):

    print(f'Calculating Payoffs for {config}')

    with open(f'{config}graph_results/env_config.json') as f:
        env_config = json.load(f)

    init_env = DynamicMultiAgentTransportationNetworkEnvironment(env_config)

    payoff_table = np.loadtxt(f'{config}graph_results/payoff_table.csv', delimiter=',')

    payoffs = []

    for i in range(1, len(payoff_table)):
        current_table = payoff_table[:i, :i]
        payoff = - get_value(current_table) * init_env.max_number_of_vehicles
        payoffs.append(payoff)

        # if i < len(payoff_table) - 1:
        #     current_table = payoff_table[:i+2, :i+1]
        #     payoff = - get_value(current_table) * init_env.max_number_of_vehicles
        #     payoffs.append(payoff)

    print(payoffs)

    plt.scatter(np.arange(len(payoffs)), payoffs)
    plt.plot(payoffs)
    plt.ylabel('Total Travel Time')
    plt.xlabel('Training Iteration')
    plt.grid(True, which='major', linestyle='-', axis='y')
    # plt.savefig(f'report_{config}.png')
    tikzplotlib.save(f'{config}graph_results/do.tikz')
    plt.clf()


def plot(config, data, labels, name):
    fig, ax = plt.subplots()
    ax.boxplot(data.T)
    ax.set_title("")
    # ax.set_xlabel("Strategies")
    ax.set_xticks(
        ticks=np.arange(len(data)) + 1,
        labels=labels,
        rotation=20,
        rotation_mode='anchor',
        va='top',
        y=-0.05,
    )
    ax.yaxis.grid(True)
    ax.yaxis.grid(True, which='minor', linestyle='--')
    ax.set_ylabel('Total Travel Time')

    ax.axvline(x=2.5, color='black', linestyle='--')

    fig.tight_layout()
    # plt.savefig(f'{config}graph_results/report_{name}.png')
    tikzplotlib.save(f'{config}graph_results/report_{name}.tikz')
    plt.clf()


if __name__ == '__main__':
    # extract_comparison('3x2')
    extract_comparison('sf')
    # extract_do_plot('3x2')
    # extract_do_plot('sf')
