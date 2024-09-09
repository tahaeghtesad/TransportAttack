import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from transport_env.GymMultiEnv import ComponentTrainer


def forward_greedy(observation):
    budget = observation[-1]
    observation = observation[:-1]
    n_edges = len(observation) // 5
    features = observation.reshape((n_edges, 5))
    action = features[:, 0] / max(sum(features[:, 0]), 1e-8) * budget
    return action


def test(index):
    env_config = dict(
        network=dict(
            method='network_file',
            city='SiouxFalls',
            randomize_factor=0.00,
        ),
        horizon=50,
        render_mode=None,
        congestion=True,
        rewarding_rule='proportional',
        reward_multiplier=1.0,
        n_components=10,
    )
    edge_component_mapping = [
        [0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, 18, 22, 30, 34],
        [12, 15, 16, 17, 19, 20, 21, 23, 24, 25, 28, 31, 42, 46, 47, 49, 50, 51, 53, 54, 59],
        [6, 9, 26, 32, 33, 35, 36, 37, 38, 39, 41, 43, 65, 69, 70, 72, 73, 75],
        [27, 29, 40, 44, 45, 48, 52, 56, 57, 58, 55, 60, 61, 62, 63, 64, 66, 67, 68, 71, 74]
    ]

    def create(horizon):
        def env_creator():
            config = env_config.copy()
            config['horizon'] = horizon
            return ComponentTrainer(config, edge_component_mapping, index)

        return env_creator

    eval_env = create(50)()
    total_reward = 0
    total_steps = 0

    for epoch in range(50):
        done = False
        truncated = False
        steps = 0
        reward_sum = 0
        obs, _ = eval_env.reset()
        while not done:
            action = forward_greedy(obs)
            obs, reward, done, truncated, info = eval_env.step(action)
            reward_sum += reward
            steps += 1
            if truncated:
                break

        total_reward += reward_sum
        total_steps += steps

    print(f'Index: {index}')
    print(f'Reward Average: {total_reward / 50}')
    print(f'Step average: {total_steps / 50}')


if __name__ == '__main__':
    for index in range(4):
        test(index)

