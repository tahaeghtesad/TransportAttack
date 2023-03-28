from datetime import datetime
from typing import Optional

import gym
import stable_baselines3 as sb
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from transport_env.NetworkEnv import TransportationNetworkEnvironment
from util.torch.gcn import GraphConvolutionLayer
from util.torch.misc import allocate_best_device

import torch


class CustomGraphConvolution(BaseFeaturesExtractor):
    def __init__(self, observation_space, conv_layers, features_dim=1, adjacency_matrix=None):
        super(CustomGraphConvolution, self).__init__(observation_space, features_dim)

        self.gcn_layers = torch.nn.ModuleList([
            GraphConvolutionLayer(conv_layers[i - 1]['size'] if i != 0 else observation_space.shape[1],
                                  conv_layers[i]['size'], adjacency_matrix)
            for i in range(len(conv_layers))
        ])
        self.gcn_layers_activation = [
            getattr(torch.nn.functional, conv_layers[i]['activation'])
            for i in range(len(conv_layers))
        ]

        self.output_layer = torch.nn.Linear(conv_layers[-1]['size'] * adjacency_matrix.shape[0], features_dim)

    def forward(self, observations):
        x = observations
        for l, a in zip(self.gcn_layers, self.gcn_layers_activation):
            x = a(l(x))

        return self.output_layer(torch.flatten(x, start_dim=1))


if __name__ == '__main__':
    config = dict(
        env_config=dict(
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
            observation_type='vector'
        ),
        model_config=dict(
            conv_layers=[
                dict(size=64, activation='relu'),
                dict(size=64, activation='relu'),
                dict(size=64, activation='relu'),
                dict(size=64, activation='relu'),
                dict(size=64, activation='relu'),
                dict(size=64, activation='relu'),
            ],
            features_dim=512,
            net_arch=dict(
                pi=[512, 512],
                vf=[256, 256]
            ),
            activation_fn=torch.nn.ReLU
        )
    )

    run_id = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    env = gym.wrappers.TimeLimit(TransportationNetworkEnvironment(config['env_config']),
                                 max_episode_steps=config['env_config']['horizon'])

    model = sb.PPO(
        'MlpPolicy',
        env,
        policy_kwargs=dict(
            features_extractor_class=CustomGraphConvolution,
            features_extractor_kwargs=dict(
                conv_layers=config['model_config']['conv_layers'],
                adjacency_matrix=env.env.get_adjacency_matrix(),
                features_dim=config['model_config']['features_dim']
            ),
            net_arch=config['model_config']['net_arch'],
            activation_fn=config['model_config']['activation_fn']
        ),
        n_steps=512,
        batch_size=32,
        n_epochs=32,
        # normalize_advantage=True,
        learning_rate=3e-5,
        verbose=1,
        gamma=0.97,
        gae_lambda=0.99,
        # ent_coef=0.01,
        # vf_coef=1.0,
        # target_kl=0.01,
        device=allocate_best_device(),
        tensorboard_log=f'logs/sb_ppo/'
    )
    model.learn(
        total_timesteps=1_000_000,
        tb_log_name=f'{run_id}',
        callback=[
            EvalCallback(
                eval_env=Monitor(gym.wrappers.TimeLimit(
                    TransportationNetworkEnvironment(config['env_config']),
                    max_episode_steps=config['env_config']['horizon']
                )),
                eval_freq=10_000,
                log_path=f'logs/sb_ppo/{run_id}_1',
            ),
            CheckpointCallback(
                save_freq=10_000,
                save_path=f'logs/sb_ppo/{run_id}_1/weights',
            )
        ]
    )
