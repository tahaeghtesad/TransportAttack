import logging
import sys
import os
import traceback
from datetime import datetime
from multiprocessing import Process, Pipe

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import util.rl.exploration
from util import visualize
from util.rl.experience_replay import ExperienceReplay
from util.visualize import Timer
from util.tf.math import r2, cdist


def get_q_model(env, config, name):
    action_shape = env.action_space.sample().shape  # (76, )
    state_shape = env.observation_space.sample().shape  # (76, 2)

    state_in = tf.keras.Input(shape=state_shape)
    # action_in = tf.keras.Input(shape=action_shape)

    for i, l in enumerate(config['model_config']['dense_layers']):
        if i == 0:
            centroid_network = tf.keras.layers.Dense(l['size'], activation=l['activation'],
                                                     name=f'centroids_network_{i}')(state_in)
        else:
            centroid_network = tf.keras.layers.Dense(l['size'], activation=l['activation'],
                                                     name=f'centroids_network_{i}')(centroid_network)
        centroid_network = tf.keras.layers.Dropout(l['dropout'], name=f'centroids_network_{i}_dropout')(centroid_network)

    centroid_network = tf.keras.layers.Dense(action_shape[0] * config['model_config']['rbvf_centroids'],
                                             activation='tanh', name='centroids_network')(centroid_network)
    centroid_network = tf.keras.layers.Reshape((config['model_config']['rbvf_centroids'], action_shape[0]),
                                               name='centroids_network_reshape')(centroid_network)

    for i, l in enumerate(config['model_config']['dense_layers']):
        if i == 0:
            value_network = tf.keras.layers.Dense(l['size'], activation=l['activation'], name=f'value_network_{i}')(
                state_in)
        else:
            value_network = tf.keras.layers.Dense(l['size'], activation=l['activation'], name=f'value_network_{i}')(
                value_network)
        value_network = tf.keras.layers.Dropout(l['dropout'], name=f'value_network_{i}_dropout')(value_network)

    value_network = tf.keras.layers.Dense(config['model_config']['rbvf_centroids'], activation='linear',
                                          name='value_network')(value_network)

    model = tf.keras.Model(inputs=state_in, outputs=[centroid_network, value_network], name=name)
    model.compile(
        loss=tf.keras.losses.get(config['model_config']['loss']),
        optimizer=tf.keras.optimizers.get(config['model_config']['optimizer']),
    )
    return model


class CAQLModel:
    def __init__(self, env, config):  # purpose can be either 'agent' or 'target'
        self.q_model = get_q_model(env, config, name='q_model')
        self.target_q_model = get_q_model(env, config, name='target_q_model')
        self.sync_weights()
        self.config = config
        self.env = env

    # @tf.function
    def rbf(self, actions_2d, centroids):
        diff_norm = cdist(centroids, actions_2d)
        diff_norm *= -1 * self.config['model_config']['beta']
        return tf.nn.softmax(diff_norm, axis=2)

    # @tf.function
    def rbf_action(self, actions, centroids):
        diff_norm = centroids - tf.expand_dims(actions, axis=1)
        diff_norm = diff_norm ** 2
        diff_norm = tf.reduce_sum(diff_norm, axis=2)
        diff_norm = tf.sqrt(diff_norm + 1e-7)
        diff_norm = -1 * self.config['model_config']['beta'] * diff_norm
        weights = tf.nn.softmax(diff_norm, axis=1)  # batch x N
        return weights

    # @tf.function
    def get_optimal_action_value(self, model, states):
        centroids, values = model(states)  # (batch_size, centroid_count, num_action), (batch_size, centroid_count)
        weights = self.rbf(centroids, centroids)  # (batch_size, centroid_count, centroid_count)
        qs = tf.squeeze(tf.matmul(weights, tf.expand_dims(values, axis=2)), axis=2)  # (batch_size, centroid_count)

        return tf.gather(centroids, tf.argmax(qs, axis=1), batch_dims=1), tf.reduce_max(qs, axis=1)

    # @tf.function
    def get_q_value(self, model, states, actions):
        centroids, values = model(states)  # (batch_size, centroid_count, num_action), (batch_size, centroid_count)
        weights = self.rbf_action(actions, centroids)  # (batch_size, action_count, centroid_count)
        qs = tf.squeeze(weights) * values  # (batch_size, centroid_count)
        return tf.reduce_sum(qs, axis=1)

    @tf.function
    def policy(self, states):
        return self.get_optimal_action_value(self.target_q_model, states)[0]

    def sync_weights(self):
        self.q_model.set_weights(self.target_q_model.get_weights())

    def summary(self, print_fn):
        self.q_model.summary(print_fn=print_fn)
        self.target_q_model.summary(print_fn=print_fn)

    # @tf.function
    def train(self, states, actions, next_states, rewards, dones):
        rewards = tf.clip_by_value(rewards, -self.config['rl_config']['reward_clip'], self.config['rl_config']['reward_clip'])
        next_actions, next_q_values = self.get_optimal_action_value(self.target_q_model, next_states)
        target_q_values = rewards + (1 - dones) * self.config['rl_config']['gamma'] * next_q_values

        with tf.GradientTape() as tape:

            current_q_values = self.get_q_value(self.q_model, states, actions)
            loss = self.q_model.compiled_loss(target_q_values, current_q_values)

        # grads = tf.gradients(loss, self.q_model.trainable_variables)
        grads = tape.gradient(loss, self.q_model.trainable_variables)
        self.q_model.optimizer.apply_gradients(zip(grads, self.q_model.trainable_variables))

        # Update Target Networks
        self.update_target_model()

        # return dict(
        #     average_actions=tf.reduce_mean(next_actions,
        #                                    axis=0),
        #     max_actions=tf.reduce_max(next_actions,
        #                                 axis=0),
        #     min_actions=tf.reduce_min(next_actions,
        #                                 axis=0),
        #     q=dict(
        #         loss=loss,
        #         r2=r2(current_q_values, target_q_values),
        #         max_q=tf.reduce_max(target_q_values),
        #         min_q=tf.reduce_min(target_q_values),
        #         mean_q=tf.reduce_mean(target_q_values)
        #     )
        # )

    @tf.function
    def update_target_model(self):
        for var, target_var in zip(self.q_model.trainable_variables, self.target_q_model.trainable_variables):
            target_var.assign(
                self.config['rl_config']['tau'] * var + (1 - self.config['rl_config']['tau']) * target_var)


class Agent(Process):
    def __init__(self, index, config, pipe) -> None:
        super().__init__(name=f'Agent-{index}')
        self.noise = None
        self.pipe = pipe
        self.config = config
        self.finished = False
        self.index = index

        self.logger = logging.getLogger(f'Agent-{index}')

        self.model = None
        self.env = None
        self.render_env = None

        self.logger.info(f'Agent {self.index} created.')

        self.obs = None
        self.done = False
        self.truncated = False
        self.epsilon = None

    def run(self) -> None:
        self.logger.info(f'Initializing Agent {self.index}')

        gpus = tf.config.list_physical_devices('GPU')
        assigned_gpu = self.index % len(gpus)
        tf.config.set_visible_devices(gpus[assigned_gpu], 'GPU')
        tf.config.set_logical_device_configuration(
            gpus[assigned_gpu],
            [tf.config.LogicalDeviceConfiguration(memory_limit=self.config['training_config']['agent_gpu_memory'])])

        self.logger.info(f'Initializing environment.')
        self.env = gym.wrappers.TimeLimit(gym.make('LunarLander-v2', continuous=True),
                                          max_episode_steps=2 * self.config['env_config']['time_limit'])
        self.obs = self.env.reset()[0]
        self.done = False
        self.truncated = False

        self.render_env = gym.wrappers.record_video.RecordVideo(
            env=gym.wrappers.TimeLimit(
                gym.make('LunarLander-v2', continuous=True, render_mode='rgb_array'),
                max_episode_steps=self.config['env_config']['time_limit']),
            video_folder=f'{self.config["training_config"]["logdir"]}/videos/',
            episode_trigger=lambda episode: True
        )
        self.logger.info(f'Initializing model.')
        self.model = CAQLModel(self.env, self.config)
        self.logger.info(f'Initializing exploration strategy.')
        self.noise = getattr(util.rl.exploration, self.config['rl_config']['noise']['type']) \
            (**self.config['rl_config']['noise']['config'], shape=self.env.action_space.sample().shape)
        self.epsilon = getattr(util.rl.exploration, self.config['rl_config']['epsilon']['type']) \
            (**self.config['rl_config']['epsilon']['config'])

        self.logger.info(f'Agent {self.index} started.')
        while not self.finished:
            message = self.pipe.recv()
            try:
                if message['type'] == 'get_trajectories':
                    self.logger.debug(f'Agent {self.index} received predict message.')
                    experiences, info = self.get_trajectories()
                    self.pipe.send(
                        dict(
                            type='trajectories',
                            experiences=experiences,
                            info=info
                        )
                    )
                if message['type'] == 'update_weights':
                    self.logger.debug(f'Agent {self.index} received update weights message.')
                    self.update_model_weights(message['weights'])
                    self.pipe.send(dict(type='weights_updated'))
                elif message['type'] == 'close':
                    self.logger.debug(f'Agent {self.index} received close message.')
                    self.finished = True
                elif message['type'] == 'test':
                    self.logger.debug(f'Agent {self.index} received test message.')
                    stat = self.test_trained_model()
                    self.pipe.send(dict(type='test', stat=stat))
                elif message['type'] == 'render':
                    self.logger.debug(f'Agent {self.index} received render message.')
                    stat = self.render()
                    self.pipe.send(dict(type='rendered', stat=stat))
            except Exception as e:
                self.logger.exception(e)
                self.pipe.send(dict(type='error', error=str(e)))

    def get_trajectories(self):
        start_time = datetime.now()
        count = 0
        rewards = 0
        discounted_rewards = 0
        experiences = []
        # action = self.heuristics[0].predict(obs)

        self.obs = self.env.reset()[0]
        self.done = False
        self.truncated = False

        while not self.done and not self.truncated:

            if self.epsilon():
                action = self.env.action_space.low + np.random.rand(self.env.action_space.shape[0]) * (
                        self.env.action_space.high - self.env.action_space.low)
            else:
                action = self.model.policy(tf.expand_dims(self.obs, axis=0))[0].numpy()

            action += self.noise()

            next_obs, reward, self.done, self.truncated, info = self.env.step(action)

            experiences.append(dict(
                state=self.obs,
                action=action,
                reward=reward,
                next_state=next_obs,
                done=self.done
            ))

            self.obs = next_obs
            count += 1
            rewards += reward
            discounted_rewards += reward * self.config['rl_config']['gamma'] ** count

        return experiences, dict(
            cumulative_reward=rewards,
            average_reward=rewards / count,
            discounted_reward=discounted_rewards,
            episode_length=count,
            noise=self.noise.get_current_noise(),
            epsilon=self.epsilon.get_current_epsilon(),
            time=datetime.now() - start_time,
            # time_report=' ~ '.join([f'{timer}: {time / iterations:.3f}(s/i)' for timer, (time, iterations) in
            #                           visualize.timer_stats.items()])
        )

    def test_trained_model(self):
        start_time = datetime.now()
        env = gym.wrappers.TimeLimit(gym.make('LunarLander-v2', continuous=True),
                                     max_episode_steps=self.config['env_config']['time_limit'])
        obs = env.reset()[0]
        done = False
        truncated = False
        count = 0
        rewards = 0
        discounted_rewards = 0
        while not done and not truncated:
            action = self.model.policy(np.expand_dims(obs, axis=0))[0].numpy()

            obs, reward, done, truncated, info = env.step(action)
            count += 1
            rewards += reward
            discounted_rewards += reward * self.config['rl_config']['gamma'] ** count

        return dict(
            cumulative_reward=rewards,
            average_reward=rewards / count,
            discounted_reward=discounted_rewards,
            episode_length=count,
            time=datetime.now() - start_time
        )

    def render(self):
        obs = self.render_env.reset()[0]
        done = False
        truncated = False
        while not done and not truncated:
            action = self.model.policy(np.expand_dims(obs, axis=0))[0].numpy()

            obs, reward, done, truncated, info = self.render_env.step(action)

    def update_model_weights(self, new_weights):
        self.model.target_q_model.set_weights(new_weights)


class Trainer(Process):

    def __init__(self, config, pipe) -> None:
        super().__init__(name='Trainer')
        self.model: CAQLModel = None
        self.env = None
        self.logger = logging.getLogger('Trainer')
        self.config = config
        self.pipe = pipe

        self.replay_buffer = ExperienceReplay(self.config['rl_config']['buffer_size'],
                                              self.config['rl_config']['batch_size'])
        self.training_step = 0

    # @tf.function
    def update_model(self):
        # tf.map_fn(lambda i: self.model.train(*self.replay_buffer.sample()), tf.range(self.config['training_config']['num_training_per_epoch'], dtype=tf.int32))
        for _ in tqdm(range(self.config['training_config']['num_training_per_epoch'])):
            self.model.train(*self.replay_buffer.sample())
        self.training_step += self.config['training_config']['num_training_per_epoch']

    def init(self):
        gpus = tf.config.list_physical_devices('GPU')
        assigned_gpu = len(gpus) - 1
        tf.config.set_visible_devices(gpus[assigned_gpu], 'GPU')
        tf.config.set_logical_device_configuration(
            gpus[assigned_gpu],
            [tf.config.LogicalDeviceConfiguration(memory_limit=self.config['training_config']['trainer_gpu_memory'])])
        self.logger.info(f'All Devices: {tf.config.list_physical_devices()}')
        self.logger.info(f'Logical Devices {tf.config.list_logical_devices()}')
        self.logger.info(f'Assigned GPU {assigned_gpu} to trainer.')

        self.logger.info(f'Initializing trainer.')
        self.logger.info(f'Initializing trainer environment variables.')
        self.env = gym.make('LunarLander-v2', continuous=True)
        self.logger.info(f'Initializing trainer model.')
        self.model = CAQLModel(self.env, self.config)
        self.model.summary(print_fn=print)
        self.logger.info(f'Trainer initialized.')

    def store_model(self):
        path = f'logs/{self.config["training_config"]["run_id"]}/weights/'
        if not os.path.exists(path):
            os.makedirs(path)
        self.logger.debug(f'Storing model to {path}')
        self.model.q_model.save_weights(path + f'q_weights-{self.training_step}.h5')
        self.model.target_q_model.save_weights(path + f'target_q_weights-{self.training_step}.h5')

    def load_model(self, run_id, step):
        self.logger.debug(f'Loading model from {run_id}')
        self.model.q_model.load_weights('logs/' + run_id + f'/weights/q_weights-{step}.h5')
        self.model.target_q_model.load_weights('logs/' + run_id + f'/weights/target_q_weights-{step}.h5')
        return self.model.target_q_model.get_weights()

    def run(self) -> None:
        self.init()
        self.logger.info(f'Starting training.')
        while True:
            msg = self.pipe.recv()
            try:
                if msg['type'] == 'update_model':
                    self.logger.debug(f'Updating model.')
                    # stats = self.update_model()
                    self.update_model()
                    # pickleable_stats = [
                    #     {k: v.numpy() if isinstance(v, tf.Tensor) else v for k, v in current.items()}
                    #     for current in stats
                    # ]
                    self.pipe.send(
                        dict(
                            type='model_updated',
                            # stats=pickleable_stats,
                            stats=[],
                            weights=self.model.target_q_model.get_weights()
                        )
                    )
                elif msg['type'] == 'add_samples':
                    self.logger.debug(f'adding samples.')
                    try:
                        self.replay_buffer.batch_add(msg['experiences'])
                        self.pipe.send(dict(type='samples_added', total_experience=self.replay_buffer.size()))
                    except Exception as e:
                        self.pipe.send(dict(type='error', error=e))
                elif msg['type'] == 'store_model':
                    self.store_model()
                elif msg['type'] == 'load_model':
                    policy_weights = self.load_model(msg['run_id'], msg['step'])
                    self.pipe.send(dict(type='loaded', policy_weights=policy_weights))
                else:
                    self.logger.error(f'Invalid Message {msg}')
            except Exception as e:
                self.logger.error(f'Error raised {e}')
                traceback.print_exception(e)
                self.pipe.send(dict(type='error', error=e))


class TFLogger(Process):

    def __init__(self, config, pipe) -> None:
        super().__init__(name='TFLogger')
        self.logger = logging.getLogger('TFLogger')
        self.logger.info(f'Initializing logger.')
        self.config = config
        self.pipe = pipe
        self.finished = False

    def run(self) -> None:

        tf.config.set_visible_devices([], 'GPU')

        file_writer = tf.summary.create_file_writer(self.config["training_config"]["logdir"] + "/metrics")
        file_writer.set_as_default()

        while not self.finished:
            msg = self.pipe.recv()
            try:
                if msg['type'] == 'log_scalar':
                    tf.summary.scalar(msg['name'], msg['value'], msg['step'])
                elif msg['type'] == 'log_histogram':
                    tf.summary.histogram(msg['name'], msg['value'], msg['step'])
                elif msg['type'] == 'log_graph':
                    tf.summary.graph(msg['graph'])
                elif msg['type'] == 'flush':
                    tf.summary.flush()
                else:
                    self.logger.error(f'Invalid Message {msg}')
            except Exception as e:
                self.logger.error(f'Error raised {e}: msg={msg}')
                self.pipe.send(dict(type='error', error=e))


class Manager:
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.logger = logging.getLogger('Manager')

        self.agents = None
        self.trainer = None
        self.trainer_pipe = None
        self.tf_logger = None
        self.tf_logger_pipe = None

        self.training_step = 0

    def start(self):
        self.agents = []
        for i in range(config['training_config']['num_agents']):
            self.logger.debug(f'Creating agent {i}.')
            parent_conn, child_conn = Pipe()
            agent = Agent(i, config, child_conn)
            self.agents.append(
                dict(
                    agent=agent,
                    pipe=parent_conn
                )
            )

        self.trainer_pipe, pipe = Pipe()
        self.trainer = Trainer(config, pipe)

        self.tf_logger_pipe, pipe = Pipe()
        self.tf_logger = TFLogger(self.config, pipe)

    def log_scalar(self, name, value, step):
        self.tf_logger_pipe.send(dict(
            type='log_scalar',
            name=name,
            value=value,
            step=step
        ))

    def log_histogram(self, name, value, step):
        self.tf_logger_pipe.send(dict(
            type='log_histogram',
            name=name,
            value=value,
            step=step
        ))

    def log_graph(self, graph):
        self.tf_logger_pipe.send(dict(
            type='log_graph',
            graph=graph
        ))

    def flush_summary(self):
        self.tf_logger_pipe.send(dict(
            type='flush',
        ))


    def killall(self):
        for agent in self.agents:
            agent['agent'].kill()
        self.trainer.kill()
        self.tf_logger.kill()

    def train(self):

        self.start()

        self.logger.info(f'Starting agents.')
        for agent in self.agents:
            agent['agent'].start()
        self.trainer.start()
        self.tf_logger.start()

        if 'resume_from' in self.config['training_config']:
            self.trainer_pipe.send(
                dict(
                    type='load_model',
                    run_id=self.config['training_config']['resume_from']['run_id'],
                    step=self.config['training_config']['resume_from']['step']
                )
            )
            ack = self.trainer_pipe.recv()
            assert ack['type'] == 'loaded', f'Model could not be loaded {ack}'
            policy_weights = ack['policy_weights']
            for agent in self.agents:
                agent['pipe'].send(dict(
                    type='update_weights',
                    weights=policy_weights
                ))

            for agent in self.agents:
                assert agent['pipe'].recv()['type'] == 'weights_updated'

        total_samples = 0
        for iteration in (pbar := tqdm(range(self.config['rl_config']['num_episodes']))):
            with Timer('GetTrajectories'):
                for i, agent in enumerate(self.agents):
                    agent['pipe'].send(dict(
                        type='get_trajectories'
                    ))
                    pbar.set_description(f'Requesting Trajectories {i}/{len(self.agents)}')
                infos = []
                for i, agent in enumerate(self.agents):
                    pbar.set_description(f'Received {i}/{len(self.agents)} trajectories')
                    msg = agent['pipe'].recv()
                    if msg['type'] == 'error':
                        raise Exception(msg['error'])

                    self.trainer_pipe.send(dict(
                        type='add_samples',
                        experiences=msg['experiences']
                    ))
                    infos.append(msg['info'])
                    total_samples += msg['info']['episode_length']
                    reply = self.trainer_pipe.recv()
                    assert reply['type'] == 'samples_added', f'Invalid reply {reply}'
                    experience_replay_buffer_size = reply['total_experience']

                self.log_scalar('rl/total_samples', total_samples, self.training_step)
                self.log_scalar('rl/experience_replay_buffer_size', experience_replay_buffer_size, self.training_step)
                self.log_scalar('rl/noise', np.average([info['noise'] for info in infos]),
                                self.training_step)
                self.log_scalar('rl/epsilon', np.average([info['epsilon'] for info in infos]),
                                self.training_step)
                self.log_scalar('env/reward', np.average([info['cumulative_reward'] for info in infos]),
                                self.training_step)
                self.log_histogram('env/episode_lengths', [info['episode_length'] for info in infos],
                                   self.training_step)
                self.log_histogram('env/cumulative_rewards', [info['cumulative_reward'] for info in infos],
                                   self.training_step)
                self.log_histogram('env/average_reward', [info['average_reward'] for info in infos],
                                   self.training_step)
                self.log_histogram('env/discounted_reward', [info['discounted_reward'] for info in infos],
                                   self.training_step)
                self.log_histogram('env/time', [info['time'].total_seconds() for info in infos],
                                   self.training_step)
                self.log_scalar('training/epoch_time', np.average([info['time'].total_seconds() for info in infos]),
                                self.training_step)
                self.flush_summary()


            with Timer('UpdateModel'):
                pbar.set_description('Updating model')
                start = datetime.now()
                self.trainer_pipe.send(
                    dict(
                        type='update_model'
                    )
                )
                ack = self.trainer_pipe.recv()
                assert ack['type'] == 'model_updated', f'Invalid reply {ack}'
                new_weights = ack['weights']

                for i, stats in enumerate(ack['stats']):
                    pass
                    # q_stats = stats['q']
                    # # policy_stats = stats['policy']
                    # average_action_0 = stats['average_actions'][0]
                    # average_action_1 = stats['average_actions'][1]
                    # min_action_0 = stats['min_actions'][0]
                    # min_action_1 = stats['min_actions'][1]
                    # max_action_0 = stats['max_actions'][0]
                    # max_action_1 = stats['max_actions'][1]
                    #
                    #
                    # self.log_scalar('env/average_action_0', average_action_0, self.training_step)
                    # self.log_scalar('env/average_action_1', average_action_1, self.training_step)
                    # self.log_scalar('env/min_action_0', min_action_0, self.training_step)
                    # self.log_scalar('env/min_action_1', min_action_1, self.training_step)
                    # self.log_scalar('env/max_action_0', max_action_0, self.training_step)
                    # self.log_scalar('env/max_action_1', max_action_1, self.training_step)
                    #
                    # self.log_scalar('rl/q_loss', q_stats['loss'], self.training_step)
                    # self.log_scalar('rl/gamma', stats['gamma'], self.training_step)
                    # self.log_scalar('rl/r2', q_stats['r2'], self.training_step)
                    # self.log_scalar('rl/max_q', q_stats['max_q'], self.training_step)
                    # self.log_scalar('rl/min_q', q_stats['min_q'], self.training_step)
                    # self.log_scalar('rl/mean_q', q_stats['mean_q'], self.training_step)
                    # self.log_scalar('constants/tau', self.config['rl_config']['tau'], self.training_step)
                    # self.log_scalar('rl/policy_loss', policy_stats['loss'], self.training_step)
                    # self.log_scalar('rl/policy_r2', policy_stats['r2'], self.training_step)

                self.training_step += self.config['training_config']['num_training_per_epoch']

                self.log_scalar('training/train_time', (datetime.now() - start).total_seconds(), self.training_step)

                pbar.set_description(f'Sending updated weights.')

                for agent in self.agents:
                    agent['pipe'].send(dict(
                        type='update_weights',
                        weights=new_weights
                    ))

                for agent in self.agents:
                    assert agent['pipe'].recv()['type'] == 'weights_updated'

                if ((self.training_step - 1) // self.config['training_config']['num_training_per_epoch']) % \
                        self.config['training_config']['checkpoint_interval'] == 0:
                    self.logger.debug(f'Checkpointing')

                    for agent in self.agents:
                        agent['pipe'].send(dict(
                            type='test'
                        ))

                    stats = []
                    for i, agent in enumerate(self.agents):
                        pbar.set_description(f'Testing {self.training_step} {i}/{len(self.agents)}.')
                        msg = agent['pipe'].recv()
                        assert msg['type'] == 'test'
                        stats.append(msg['stat'])
                    print('done testing')

                    self.log_scalar('env/test_reward', np.average([stat['cumulative_reward'] for stat in stats]),
                                    self.training_step)

                    pbar.set_description('Rendering...')

                    self.agents[0]['pipe'].send(dict(
                        type='render'
                    ))
                    ack = self.agents[0]['pipe'].recv()
                    assert ack['type'] == 'rendered', f'Failed to render. {ack}'

                    self.trainer_pipe.send(dict(
                        type='store_model'
                    ))

            time_report = ' ~ '.join([f'{timer}: {time / iterations:.3f}(s/i)' for timer, (time, iterations) in
                                      visualize.timer_stats.items()])
            # pbar.set_description(f'{time_report}')


if __name__ == '__main__':
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    config = dict(
        env_config=dict(
            time_limit=256
        ),
        model_config=dict(
            optimizer=dict(
                class_name='RMSProp',
                config=dict(
                    learning_rate=0.000001
                )
            ),
            rbvf_centroids=128,
            loss='MeanSquaredError',
            dense_layers=[
                dict(size=512, activation='relu', dropout=0.4),
                dict(size=512, activation='relu', dropout=0.4),
                dict(size=512, activation='relu', dropout=0.4)
            ],
            beta=2.0,
        ),
        rl_config=dict(
            epsilon=dict(
                type='DecayEpsilon',
                config=dict(
                    epsilon_start=1.0,
                    epsilon_end=0.0,
                    epsilon_decay=10_000
                )
            ),
            # noise=dict(
            #     type='OUActionNoise',
            #     config=dict(
            #         theta=0.15,
            #         mean=0,
            #         std_deviation=0.3,
            #         dt=0.01,
            #         target_scale=0.01,
            #         anneal=200_000
            #     )
            # ),
            noise=dict(
                type='ZeroNoise',
                config=dict()
            ),
            reward_clip=20,
            tau=0.005,
            gamma=0.99,
            batch_size=256,
            buffer_size=500_000,
            num_episodes=10000
        ),
        training_config=dict(
            num_agents=1,
            num_training_per_epoch=1024,
            run_id=run_id,
            agent_gpu_memory=512,
            trainer_gpu_memory=512,
            logdir=f'logs/{run_id}',
            checkpoint_interval=1,
            # resume_from=dict(
            #     run_id='20230214-154600',
            #     step=2248
            # )
        ),
    )

    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s',
                        level=logging.INFO
                        # level=0
                        )

    logger = logging.getLogger(__name__)
    logger.info(f'Starting run {run_id}...')

    manager = Manager(config)
    try:
        manager.train()
    except BaseException as e:
        logger.exception(f'Exception {e}')
    manager.killall()