from typing import Tuple, Callable

import gymnasium as gym
import numpy as np
import torch.nn
from gymnasium.vector.utils import spaces
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from transport_env.GymMultiEnv import GymMultiEnv, MultiEnvWithDecoder, MultiEnvWithDecoderAndDetector


class BatchedGraphConvolution(torch.nn.Module):

    def __init__(self, adj, n_edges, input_dim, output_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.n_edges = n_edges
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.register_buffer('adj', torch.eye(n_edges) + torch.from_numpy(adj))
        self.register_parameter('weight', torch.nn.Parameter(torch.randn((n_edges, input_dim, output_dim)), requires_grad=True))

    def forward(self, x):
        # Shape of x: (batch, n_edges, input_dim)
        batch_size = x.size(0)

        # Apply the adjacency matrix
        # adj has shape (n_edges, n_edges)
        x = torch.einsum('ij,bjd->bid', self.adj, x)  # x now has shape (batch, n_edges, input_dim)

        # Multiply each (n_edges, input_dim) slice with the corresponding (input_dim, output_dim) weight slice
        # weight has shape (n_edges, input_dim, output_dim)
        output = torch.einsum('bni,nio->bno', x, self.weight)  # output has shape (batch, n_edges, output_dim)

        return output


class CustomGNN(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.Space, adj, features_dim: int = 0) -> None:
        super().__init__(observation_space, features_dim)
        self.model = torch.nn.Sequential(
            BatchedGraphConvolution(adj, 76, 5, 32),
            torch.nn.Tanh(),
            BatchedGraphConvolution(adj, 76, 32, 32),
            torch.nn.Tanh(),
            BatchedGraphConvolution(adj, 76, 32, features_dim),
            torch.nn.Tanh(),
            torch.nn.Flatten(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.model(observations)

    @property
    def features_dim(self) -> int:
        return super().features_dim * 76


if __name__ == '__main__':

    n_envs = 128
    env_config = dict(
        network=dict(
            method='network_file',
            city='SiouxFalls',
            randomize_factor=0.00,
        ),
        horizon=200,
        render_mode=None,
        congestion=True,
        rewarding_rule='proportional',
        reward_multiplier=1.0,
        n_components=4,
    )

    edge_component_mapping = [
        [0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, 18, 22, 30, 34],
        [12, 15, 16, 17, 19, 20, 21, 23, 24, 25, 28, 31, 42, 46, 47, 49, 50, 51, 53, 54, 59],
        [6, 9, 26, 32, 33, 35, 36, 37, 38, 39, 41, 43, 65, 69, 70, 72, 73, 75],
        [27, 29, 40, 44, 45, 48, 52, 56, 57, 58, 55, 60, 61, 62, 63, 64, 66, 67, 68, 71, 74]
    ]

    def create(horizon):
        def f():
            config = env_config.copy()
            config['horizon'] = horizon
            return MultiEnvWithDecoderAndDetector(config, edge_component_mapping)
        return f

    env = make_vec_env(create(50), n_envs=n_envs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))
    eval_env = make_vec_env(create(50), n_envs=1, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))
    # env = VecNormalize(env)

    model = PPO(
        "MlpPolicy",
        # policy_kwargs=dict(
            # net_arch=dict(pi=[128, 128], vf=[128, 128, 128]),
            # features_extractor_class=CustomGNN,
            # features_extractor_kwargs=dict(adj=env.get_attr('get_adj', [0])[0](), features_dim=1),
        # ),
        env=env,
        verbose=2,
        ent_coef=0.01,
        tensorboard_log="./logs-sb",
        n_steps=50,
        device='cuda:1'
    )
    model.learn(total_timesteps=5_000_000, progress_bar=True,
                callback=[
                    EvalCallback(eval_env, n_eval_episodes=50, eval_freq=1000,
                                 callback_on_new_best=CheckpointCallback(
                                     save_freq=1, save_path='saved_partial_models', name_prefix=f'ppo_with_decoder'
                                 )
                                 )
                ])
    model.save('ppo')