from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from transport_env.GymMultiEnv import ComponentTrainerWithDecoder, ComponentTrainer


def run(index):
    n_envs = 128
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
        n_components=4,
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
            return ComponentTrainerWithDecoder(config, edge_component_mapping, index)

        return env_creator

    env = make_vec_env(create(50), n_envs=n_envs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))
    eval_env = make_vec_env(create(50), n_envs=n_envs, vec_env_cls=SubprocVecEnv,
                            vec_env_kwargs=dict(start_method='fork'))
    # env = VecNormalize(env)

    model = PPO(
        "MlpPolicy",
        env=env,
        verbose=0,
        tensorboard_log=f"./logs-sb-{index}",
        n_steps=50,
        device='cuda:1'
    )
    model.learn(total_timesteps=1_000_000, progress_bar=True,
                callback=[
                    EvalCallback(eval_env, n_eval_episodes=100, eval_freq=1000, verbose=0,
                                 # callback_on_new_best=CheckpointCallback(
                                 #     save_freq=1, save_path='saved_partial_models', name_prefix=f'ppo_{index}'
                                 # )
                                 )
                ])
    model.save(f'saved_partial_models/ppo_{index}_final')


if __name__ == '__main__':
    for i in range(4):
        run(i)

