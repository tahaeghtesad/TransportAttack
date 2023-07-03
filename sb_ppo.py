from datetime import datetime
from typing import Tuple, Optional, Union

import gym
import stable_baselines3 as sb
import torch
import torch as th
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.distributions import Distribution, SelfDistribution
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from torch import nn

from transport_env.NetworkEnv import TransportationNetworkEnvironment
from util.torch.gcn import GraphConvolutionResidualBlock
from util.torch.misc import allocate_best_device


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
        self.output_layer = torch.nn.Linear(observation_space.shape[0] * observation_space.shape[1], features_dim)

    def forward(self, observations):
        x = observations
        for l in self.gcn_layers:
            x = l(x)

        return self.output_layer(torch.flatten(x, start_dim=1))


class NormalizingActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            net_arch=None,
            activation_fn=torch.nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class=FlattenExtractor,
            features_extractor_kwargs=None,
            share_features_extractor: bool = True,
            normalize_images: bool = True,
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs=None,
            norm=2, epsilon=30
    ):
        super(NormalizingActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs
        )
        self.norm = norm
        self.epsilon = epsilon

    def forward(self, obs, deterministic: bool = False):
        actions, values, log_prob = super().forward(obs, deterministic)
        return torch.nn.functional.normalize(actions, p=self.norm) * self.epsilon, values, log_prob

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        actions = super()._predict(observation, deterministic)
        return torch.nn.functional.normalize(actions, p=self.norm) * self.epsilon


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
        share_features_extractor=True,
        rewarding_rule='vehicle_count',
        observation_type='vector',
        norm_penalty_coeff=0.005,
    )

    run_id = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    env = Monitor(
            TransportationNetworkEnvironment(env_config),
    )

    model = sb.PPO(
        NormalizingActorCriticPolicy,
        env,
        policy_kwargs=dict(
            features_extractor_class=CustomGraphConvolution,
            features_extractor_kwargs=dict(
                conv_layers=[
                    dict(depth=3, activation='relu'),
                    dict(depth=3, activation='relu'),
                    dict(depth=3, activation='relu'),
                ],
                adjacency_matrix=env.env.get_adjacency_matrix(),
                features_dim=256,
            ),
            net_arch=dict(
                pi=[512, 512],
                vf=[512, 512]
            ),
            activation_fn=torch.nn.ReLU,
            norm=env_config['norm'],
            epsilon=env_config['epsilon']
        ),
        # n_steps=128,
        # batch_size=16,
        # n_epochs=1,
        # learning_rate=3e-4,
        verbose=2,
        gamma=0.97,
        ent_coef=0.05,
        # vf_coef=0.1,
        # target_kl=0.1,
        # use_sde=True,
        # sde_sample_freq=1,
        device=allocate_best_device(),
        tensorboard_log=f'logs/sb_ppo/'
    )

    print(model.policy)

    model.learn(
        total_timesteps=10_000_000,
        tb_log_name=f'{run_id}',
        callback=[
            EvalCallback(
                eval_env=env,
                eval_freq=1_000,
                log_path=f'logs/sb_ppo/{run_id}_1',
                verbose=0,
            ),
            CheckpointCallback(
                save_freq=1_000,
                save_path=f'logs/sb_ppo/{run_id}_1/weights',
                verbose=0,
            ),
            TensorboardCallback()
        ]
    )
