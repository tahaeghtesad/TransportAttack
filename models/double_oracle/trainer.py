import os

import numpy as np
from rich.console import Console
from rich.progress import Progress
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from models import CustomModule
from strategies.attacker_strategies import BaseAttackerStrategy, MixedAttackerStrategy, ZeroAttackStrategy
from strategies.defender_strategies import BaseDefenderStrategy, MixedDefenderStrategy
from transport_env.AdvEnv import BasicAttackerEnv, BasicDefenderEnv, BaseAdvEnv
from util.math import solve_lp
from util.rl.history_environment import HistoryEnvironment


class Trainer(CustomModule):

    def __init__(self, env_config, config) -> None:
        super().__init__('DOTrainer')
        self.env_config = env_config
        self.config = config

        self.attacker_strategy_sets: list[BaseAttackerStrategy] = []
        self.defender_strategy_sets: list[BaseDefenderStrategy] = []

        # Rows are the defender, columns are the attacker. Values are the payoffs for the defender.
        self.payoff_table = []

    def train_attacker(self, probabilities) -> str:  # returns attacker strategy
        index = len(probabilities)
        model_name = f'ppo_attacker_{index}'
        if os.path.exists(f'{model_name}.zip'):
            return model_name

        def create(horizon):
            def f():
                config = self.env_config.copy()
                config['horizon'] = horizon
                return BasicAttackerEnv(config, defender_strategy=MixedDefenderStrategy(self.defender_strategy_sets, probabilities))

            return f

        env = make_vec_env(create(50), n_envs=self.config['n_envs'], vec_env_cls=SubprocVecEnv,
                           vec_env_kwargs=dict(start_method='fork'))
        eval_env = make_vec_env(create(50), n_envs=1, vec_env_cls=SubprocVecEnv,
                                vec_env_kwargs=dict(start_method='fork'))
        # env = VecNormalize(env)

        model = PPO(
            "MlpPolicy",
            # policy_kwargs=dict(
            #     # net_arch=dict(pi=[128, 128], vf=[128, 128, 128]),
            #     # features_extractor_class=CustomGNN,
            #     # features_extractor_kwargs=dict(adj=env.get_attr('get_adj', [0])[0](), features_dim=1),
            #     squash_output=True
            # ),
            env=env,
            verbose=2,
            ent_coef=0.01,
            tensorboard_log="./logs-sb-attacker",
            n_steps=50,
            device=self.device
        )
        model.learn(total_timesteps=self.config['attacker_training_steps'], progress_bar=True,
                    callback=[
                        EvalCallback(eval_env, n_eval_episodes=50, eval_freq=1000,
                                     callback_on_new_best=CheckpointCallback(
                                         save_freq=1, save_path=f'saved_partial_models',
                                         name_prefix=model_name
                                     )
                                     )
                    ])
        model.save(model_name)
        return model_name

    def train_detector(self, probabilities) -> str:  # returns defender strategy
        index = len(probabilities)
        model_name = f'ppo_defender_{index}'

        if os.path.exists(f'{model_name}.zip'):
            return model_name

        def create(horizon):
            def f():
                config = self.env_config.copy()
                config['horizon'] = horizon
                return HistoryEnvironment(BasicDefenderEnv(config, MixedAttackerStrategy(
                    [ZeroAttackStrategy(), MixedAttackerStrategy(
                        self.attacker_strategy_sets,
                        probabilities
                    )],
                    [0.5, 0.5]
                )), self.config['defender_n_history'])

            return f

        env = make_vec_env(create(50), n_envs=self.config['n_envs'], vec_env_cls=SubprocVecEnv,
                           vec_env_kwargs=dict(start_method='fork'))
        eval_env = make_vec_env(create(50), n_envs=1, vec_env_cls=SubprocVecEnv,
                                vec_env_kwargs=dict(start_method='fork'))

        model = PPO(
            "MlpPolicy",
            # policy_kwargs=dict(
            #     # net_arch=dict(pi=[128, 128], vf=[128, 128, 128]),
            #     # features_extractor_class=CustomGNN,
            #     # features_extractor_kwargs=dict(adj=env.get_attr('get_adj', [0])[0](), features_dim=1),
            #     squash_output=True
            # ),
            env=env,
            verbose=2,
            ent_coef=0.01,
            tensorboard_log="./logs-sb-defender",
            n_steps=50,
            device=self.device
        )
        model.learn(total_timesteps=self.config['defender_training_steps'], progress_bar=True,
                    callback=[
                        EvalCallback(eval_env, n_eval_episodes=50, eval_freq=1000,
                                     callback_on_new_best=CheckpointCallback(
                                         save_freq=1, save_path='saved_partial_models',
                                         name_prefix=model_name
                                     )
                                     )
                    ])
        model.save(model_name)
        return model_name

    def play_all(self, attacker_model: BaseAttackerStrategy, defender_model: BaseDefenderStrategy) -> list[float]:

        # assert not isinstance(attacker_model, MixedAttackerStrategy)
        # assert not isinstance(defender_model, MixedDefenderStrategy)

        global_step = 0
        detected_episodes = 0
        episode_rewards = []

        env = HistoryEnvironment(BasicDefenderEnv(self.env_config, attacker_model, penalty=0.0), n_history=self.config['defender_n_history'])

        for episode in range(self.config['do_config']['testing_epochs']):

            defender_model.reset()

            done = False
            truncated = False
            obs, _ = env.reset()
            step_count = 0

            episode_reward = 0

            while not done and not truncated:
                step_count += 1
                global_step += 1

                defender_action = defender_model.predict(obs, env.previous_attacker_action)
                obs, rewards, done, truncated, info = env.step(defender_action)

                episode_reward -= rewards

            episode_rewards.append(episode_reward)

        return episode_rewards

    def play(self, attacker_model: BaseAttackerStrategy, defender_model: BaseDefenderStrategy) -> float:
        return np.mean(self.play_all(attacker_model, defender_model))

    def get_attacker_payoff(self, attacker: BaseAttackerStrategy):  # returns a list of payoffs for the defender
        payoffs = []

        for i, defender in enumerate(self.defender_strategy_sets):
            payoff = self.play(attacker, defender)
            payoffs.append(payoff)

        return payoffs

    def get_defender_payoff(self, defender: BaseDefenderStrategy):  # returns a list of payoffs for the attacker
        payoffs = []
        for i, attacker in enumerate(self.attacker_strategy_sets):
            payoff = self.play(attacker, defender)
            payoffs.append(payoff)

        return payoffs

    def store_payoff_table(self):
        os.makedirs(f'logs/{self.config["run_id"]}', exist_ok=True)
        np.savetxt(f'logs/{self.config["run_id"]}/payoff_table.csv', np.array(self.payoff_table), delimiter=',')

    def append_attacker_payoffs(self, attacker_payoffs: list):
        assert len(self.defender_strategy_sets) == len(attacker_payoffs)
        for i, payoff in enumerate(attacker_payoffs):
            self.payoff_table[i].append(payoff)

    def append_defender_payoffs(self, defender_payoffs: list):
        assert len(self.attacker_strategy_sets) == len(defender_payoffs)
        self.payoff_table.append(defender_payoffs)

    # use solve_lp
    # input: value are gains for row player
    # output: optimal mixed strategy for column player
    # my payoff table rows are defender, columns are attacker. Values are the attacker gains.

    def solve_attacker(self):
        return solve_lp(-np.array(self.payoff_table))

    def solve_defender(self):
        return solve_lp(np.transpose(np.array(self.payoff_table)))

    def get_value(self):
        return self.solve_defender() @ -np.array(self.payoff_table) @ self.solve_attacker()
