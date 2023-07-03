from datetime import datetime
from typing import Optional, Union, List, Dict, Type, Any

import gym
import numpy as np
import stable_baselines3 as sb
import torch
import torch as th
from gym import spaces
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.td3.policies import TD3Policy
from torch import nn

from transport_env.NetworkEnv import TransportationNetworkEnvironment
from util.torch.gcn import GraphConvolutionResidualBlock
from util.torch.misc import allocate_best_device


class CustomGraphConvolution(BaseFeaturesExtractor):
    def __init__(self, observation_space, conv_layers, features_dim=1, adjacency_matrix=None):
        super(CustomGraphConvolution, self).__init__(observation_space, features_dim)

        self.gcn_layers = torch.nn.ModuleList([
            GraphConvolutionResidualBlock(
                observation_space.shape[1],
                adjacency_matrix,
                conv_layers[i]['activation'],
                conv_layers[i]['depth']
            )
            for i in range(len(conv_layers))
        ])

        self.output_layer = torch.nn.Linear(observation_space.shape[1] * adjacency_matrix.shape[0], features_dim)

    def forward(self, observations):
        x = observations
        for l in self.gcn_layers:
            x = l(x)

        return self.output_layer(torch.flatten(x, start_dim=1))


class NormalizingTD3Policy(TD3Policy):

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space, lr_schedule: Schedule,
                 net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None, normalize_images: bool = True,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None, n_critics: int = 2,
                 share_features_extractor: bool = False, squash_output: bool = False,
                 norm=2, epsilon=30):
        super().__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn,
                         features_extractor_class, features_extractor_kwargs, normalize_images, optimizer_class,
                         optimizer_kwargs, n_critics, share_features_extractor)
        self.norm = norm
        self.epsilon = epsilon
        self._squash_output = squash_output  # We should not squash the output, because we want to normalize it

    def forward(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self._predict(observation, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        action = super()._predict(observation, deterministic)
        relued_action = torch.maximum(action, torch.zeros_like(action))
        normalized_action = torch.nn.functional.normalize(relued_action, p=self.norm, dim=1) * self.epsilon
        return normalized_action


class TensorboardCallback(sb.common.callbacks.BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.tensorboard_log = None

    def _on_step(self) -> bool:
        self.logger.record(
            'env/norm_penalty',
            sum(self.locals['env'].envs[0].norm_sum) / len(self.locals['env'].envs[0].norm_sum)
        )
        return True


if __name__ == '__main__':
    env_config = dict(
        city='SiouxFalls',
        horizon=50,
        epsilon=30,
        norm=2,
        frac=0.5,
        num_sample=20,
        render_mode=None,
        reward_multiplier=1.0,
        congestion=True,
        trips=dict(
            type='demand_file',
            demand_file='Sirui/traffic_data/sf_demand.txt',
            strategy='random',
            count=10
        ),
        rewarding_rule='vehicle_count',
        observation_type='vector',
        norm_penalty_coeff=0.0,
    )

    run_id = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    env = Monitor(
        TransportationNetworkEnvironment(env_config)
    )

    model = sb.DDPG(
        NormalizingTD3Policy,
        # TD3Policy,
        env,
        policy_kwargs=dict(
            features_extractor_class=CustomGraphConvolution,
            features_extractor_kwargs=dict(
                conv_layers=[
                    dict(activation='relu', depth=3),
                    dict(activation='relu', depth=3),
                ],
                adjacency_matrix=env.get_adjacency_matrix(),
                features_dim=128
            ),
            # net_arch=dict(
            #     pi=[256, 256],
            #     qf=[256, 25]
            # ),
            activation_fn=torch.nn.ReLU,
            norm=env_config['norm'],
            epsilon=env_config['epsilon'],
            # share_features_extractor=True,
            squash_output=False
        ),
        learning_rate=2.5e-5,
        verbose=2,
        gamma=0.97,
        gradient_steps=64,
        # action_noise=OrnsteinUhlenbeckActionNoise(
        #     mean=np.zeros_like(env.action_space.sample()),
        #     sigma=0.5 * np.ones_like(env.action_space.sample())
        # ),
        # action_noise=NormalActionNoise(
        #     mean=np.zeros_like(env.action_space.sample()),
        #     sigma=0.3 * np.ones_like(env.action_space.sample())
        # ),
        device=allocate_best_device(),
        tensorboard_log=f'logs/sb_ddpg/'
    )

    print(model.policy)

    model.learn(
        total_timesteps=10_000_000,
        tb_log_name=f'{run_id}',
        callback=[
            EvalCallback(
                eval_env=env,
                eval_freq=1_000,
                log_path=f'logs/sb_ddpg/{run_id}_1',
                verbose=0,
            ),
            CheckpointCallback(
                save_freq=1_000,
                save_path=f'logs/sb_ddpg/{run_id}_1/weights',
                verbose=0,
            ),
            TensorboardCallback(

            )
        ]
    )
