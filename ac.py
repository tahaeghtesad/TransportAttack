import logging
import math
import random
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay
from tqdm import tqdm

from attack_heuristics import Random
from util.tf.gcn import GraphConvolutionLayer
from util.tf.normalizers import LNormOptimizerLayer
from transport_env.NetworkEnv import TransportationNetworkEnvironment
from transport_env.model import Trip
from util import visualize
from util.visualize import Timer


class CustomActorCritic:
    def __init__(self, env, rl_config, heuristics):
        self.env = env
        self.rl_config = rl_config

        self.logger = logging.getLogger(self.__class__.__name__)

        self.heuristics = heuristics

        self.model = self.get_ac_model(
            env,
            tf.keras.optimizers.Adam(
                learning_rate=ExponentialDecay(
                    initial_learning_rate=rl_config['learning_rate']['actor']['learning_rate_initial'],
                    decay_steps=rl_config['learning_rate']['actor']['learning_rate_decay_steps'],
                    decay_rate=rl_config['learning_rate']['actor']['learning_rate_decay'],
                )
            ),
            tf.keras.optimizers.Adam(
                learning_rate=ExponentialDecay(
                    initial_learning_rate=rl_config['learning_rate']['critic']['learning_rate_initial'],
                    decay_steps=rl_config['learning_rate']['critic']['learning_rate_decay_steps'],
                    decay_rate=rl_config['learning_rate']['critic']['learning_rate_decay'],
                )
            ),
        )
        if 'epsilon' in rl_config:
            self.epsilon = ConstantEpsilon(rl_config['epsilon'])
        elif 'decay_start' in rl_config and 'decay_end' in rl_config and 'decay_steps' in rl_config:
            self.epsilon = DecayEpsilon(rl_config['decay_start'], rl_config['decay_end'], rl_config['decay_steps'])
        else:
            raise ValueError('No epsilon decay strategy specified')
        self.replay_buffer = ExperienceReplay(rl_config['buffer_size'], rl_config['batch_size'])

        self.training_step = 0

    def update_model(self, samples):
        states = samples['states']
        actions = samples['actions']
        rewards = samples['rewards']
        next_states = samples['next_states']
        dones = samples['dones']
        next_actions = samples['next_actions']

        actor, critic = self.model

        # Update critic
        with tf.GradientTape() as tape:
            q_values = critic([next_states, actor(next_states)])
            current_val = critic([states, actions], training=True)
            target_val = rewards + self.rl_config['gamma'] * q_values * (1 - dones)
            loss = critic.loss(target_val, current_val)

        grads = tape.gradient(loss, critic.trainable_variables)
        critic.optimizer.apply_gradients(zip(grads, critic.trainable_variables))
        r2 = tfa.metrics.RSquare()
        r2.update_state(target_val, current_val)

        with tf.GradientTape() as tape:
            actor_loss = -tf.reduce_mean(critic([states, actor(states)]))

        grads = tape.gradient(actor_loss, actor.trainable_variables)
        actor.optimizer.apply_gradients(zip(grads, actor.trainable_variables))

        tf.summary.scalar('rl/q_loss', data=loss, step=self.training_step)
        tf.summary.scalar('rl/critic_lr', data=critic.optimizer._decayed_lr(tf.float32), step=self.training_step)
        tf.summary.scalar('rl/actor_lr', data=actor.optimizer._decayed_lr(tf.float32), step=self.training_step)
        tf.summary.scalar('rl/r2', data=r2.result(), step=self.training_step)

        tf.summary.scalar('rl/gamma',
                          data=self.rl_config['gamma'],
                          step=self.training_step)

    def get_ac_model(self, env, actor_optimizer, critic_optimizer):
        action_shape = env.action_space.sample().shape  # 76
        state_shape = env.observation_space.sample().shape  # (76, 2)
        adj = env.get_adjacency_matrix()

        # Critic

        state_in_c = tf.keras.layers.Input(shape=state_shape)
        action_in_c = tf.keras.layers.Input(shape=action_shape)
        action_reshaped = tf.keras.layers.Reshape((action_shape[0], 1))(action_in_c)

        mix = tf.keras.layers.Concatenate(axis=2)([state_in_c, action_reshaped])

        shared_c = GraphConvolutionLayer(8, adj, activation='elu')(mix)
        shared_c = GraphConvolutionLayer(8, adj, activation='elu')(shared_c)
        shared_c = GraphConvolutionLayer(8, adj, activation='elu')(shared_c)
        shared_c = GraphConvolutionLayer(8, adj, activation='elu')(shared_c)

        shared_c = tf.keras.layers.Flatten()(shared_c)
        shared_c = tf.keras.layers.Dense(8, activation='relu')(shared_c)

        output = tf.keras.layers.Dense(1)(shared_c)
        critic = tf.keras.Model(inputs=[state_in_c, action_in_c], outputs=output)

        critic.compile(
            optimizer=critic_optimizer,
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tfa.metrics.RSquare()]
        )

        critic.summary()

        # Actor
        state_in_a = tf.keras.layers.Input(shape=state_shape)
        shared_a = GraphConvolutionLayer(8, adj, activation='elu')(state_in_a)
        shared_a = GraphConvolutionLayer(8, adj, activation='elu')(shared_a)
        shared_a = GraphConvolutionLayer(8, adj, activation='elu')(shared_a)
        shared_a = GraphConvolutionLayer(8, adj, activation='elu')(shared_a)

        shared_a = tf.keras.layers.Flatten()(shared_a)
        shared_a = tf.keras.layers.Dense(8, activation='relu')(shared_a)

        shared_a = tf.keras.layers.Dense(action_shape[0], activation='relu')(shared_a)
        output_a = LNormOptimizerLayer(env.config['norm'], env.config['epsilon'])(shared_a)

        actor = tf.keras.Model(inputs=state_in_a, outputs=output_a)

        actor.compile(
            optimizer=actor_optimizer,
            loss=tf.keras.losses.MeanSquaredError()
        )

        actor.summary()

        return actor, critic

    def get_trajectories(self):
        obs = self.env.reset()
        done = False
        count = 0
        rewards = 0
        discounted_rewards = 0
        # action = self.heuristics[0].predict(obs)
        while not done:
            if self.epsilon():
                action = np.random.choice(self.heuristics).predict(obs)
            else:
                with Timer('model.predict'):
                    action = self.model[0](np.expand_dims(obs, axis=0))[0]

            next_obs, reward, done, _ = self.env.step(action)
            next_action = self.heuristics[0].predict(next_obs)

            # Only if it is not the last step due to time limit
            if not done or (env.time_step < env.config['horizon']):
                self.replay_buffer.add(obs, action, reward, next_obs, next_action, done)
            obs = next_obs
            action = next_action
            count += 1
            rewards += reward
            discounted_rewards += reward * self.rl_config['gamma'] ** count

        tf.summary.scalar('env/cumulative_reward', data=rewards, step=self.training_step)
        tf.summary.scalar('env/average_reward', data=rewards / count, step=self.training_step)
        tf.summary.scalar('env/discounted_reward', data=discounted_rewards, step=self.training_step)
        tf.summary.scalar('env/episode_length', data=count, step=self.training_step)
        tf.summary.scalar('rl/replay_buffer_size', data=self.replay_buffer.size(), step=self.training_step)

        return count, rewards / count

    def train(self):
        total_samples = 0
        for _ in (pbar := tqdm(range(rl_config['num_episodes']))):
            with Timer('GetTrajectories'):
                length, reward = self.get_trajectories()
            total_samples += length
            if self.replay_buffer.size() > rl_config['batch_size']:
                with Timer('UpdateModel'):
                    for _ in range(self.rl_config['num_training_per_epoch']):
                        self.update_model(self.replay_buffer.sample())
                        self.training_step += 1

            tf.summary.scalar('rl/epsilon', data=self.epsilon.get_current_epsilon(), step=self.training_step)
            tf.summary.scalar('rl/total_samples', data=total_samples, step=self.training_step)

            time_report = ' ~ '.join([f'{timer}: {time / iterations:.3f}(s/i)' for timer, (time, iterations) in
                                      visualize.timer_stats.items()])
            pbar.set_description(f'{time_report}')

    def store_model(self):
        self.logger.info(f'Storing model to {self.rl_config["logdir"]}/model/weights')
        self.model[0].save_weights(f'{self.rl_config["logdir"]}/model/actor_weights')
        self.model[1].save_weights(f'{self.rl_config["logdir"]}/model/critic_weights')

    def load_model(self, run_id):
        self.logger.info(f'Loading model from logs/{run_id}/model/weights')
        self.model[0].load_weights(f'logs/{run_id}/model/actor_weights')
        self.model[1].load_weights(f'logs/{run_id}/model/critic_weights')


class ExperienceReplay:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = []

    def add(self, obs, action, reward, next_obs, next_action, done):
        self.buffer.append(
            dict(
                state=obs,
                action=action,
                reward=reward,
                next_state=next_obs,
                next_action=next_action,
                done=done
            )
        )
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def size(self):
        return len(self.buffer)

    def sample(self):
        ret = dict(
            states=[],
            actions=[],
            rewards=[],
            next_states=[],
            dones=[],
            next_actions=[]
        )

        experiences = random.choices(self.buffer, k=self.batch_size)

        for e in experiences:
            ret['states'].append(e['state'])
            ret['actions'].append(e['action'])
            ret['rewards'].append([e['reward']])
            ret['next_states'].append(e['next_state'])
            ret['dones'].append([e['done']])
            ret['next_actions'].append(e['next_action'])

        return {k: tf.convert_to_tensor(v, dtype=tf.float32) for k, v in ret.items()}


class DecayEpsilon:
    def __init__(self, epsilon_start, epsilon_end, epsilon_decay):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.step = 0

    def reset(self):
        self.step = 0

    def __call__(self):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -1. * self.step / self.epsilon_decay)
        self.step += 1
        return random.random() < epsilon

    def get_current_epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -1. * self.step / self.epsilon_decay)


class ConstantEpsilon:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self):
        return random.random() < self.epsilon

    def get_current_epsilon(self):
        return self.epsilon


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s',
                        level=logging.INFO
                        # level=0
                        )

    logger = logging.getLogger(__name__)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = f'logs/{run_id}'
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()
    logger.info(f'Starting run {run_id}...')

    # Configuration with which the environment never terminates:
    # 1-norm: epsilon >= 11
    # 2-norm: epsilon >= 7
    # inf-norm: epsilon >= 4

    config = dict(
        city='SiouxFalls',
        horizon=50,
        epsilon=11,
        norm=1,
        frac=0.5,
        num_sample=20,
        render_mode=None,
        reward_multiplier=1.0,
        congestion=True,
    )

    rl_config = dict(
        buffer_size=5_000,
        batch_size=128,
        decay_start=1,
        decay_end=0.1,
        decay_steps=1000,
        # epsilon=0.1,
        num_episodes=10000,
        gamma=0.99,  # Probably a 0.5
        # learning_rate=0.05,
        learning_rate=dict(
            actor=dict(
                learning_rate_initial=0.001,
                learning_rate_decay=0.95,
                learning_rate_decay_steps=200,
            ),
            critic=dict(
                learning_rate_initial=0.0001,
                learning_rate_decay=0.95,
                learning_rate_decay_steps=200,
            )
        ),
        action_gradient_step=0.001,
        action_gradient_step_count=7,
        num_training_per_epoch=4,
        logdir=logdir,
    )


    logger.info(f'Config: {config}')
    logger.info(f'RL Config: {rl_config}')

    config.update(dict(trip_generator=Trip.using_demand_file(f'Sirui/traffic_data/sf_demand.txt', 'random', 100)))
    env = TransportationNetworkEnvironment(config)

    heuristics = [
        # GreedyRiderVector(env),
        Random(env.action_space, config['norm'], config['epsilon'], config['frac'], 'discrete'),
        # Zero(env.action_space),
    ]
    agent = CustomActorCritic(env, rl_config, heuristics=heuristics)
    # agent.load_model('20221007-144413')

    try:
        agent.train()
    except KeyboardInterrupt:
        logger.info('Interrupted by user. Exiting gracefully.')
        agent.store_model()
