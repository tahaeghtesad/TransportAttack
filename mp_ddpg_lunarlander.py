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
from util.tf.math import r2


def get_q_model(env, config, name):
    action_shape = env.action_space.sample().shape  # (76, )
    state_shape = env.observation_space.sample().shape  # (76, 2)

    state_in = tf.keras.Input(shape=state_shape)
    action_in = tf.keras.Input(shape=action_shape)

    concatenate = tf.keras.layers.Concatenate(axis=1)([state_in, action_in])
    flattened = tf.keras.layers.Flatten()(concatenate)
    for l in config['model_config']['dense_layers']:
        flattened = tf.keras.layers.Dense(l['size'], activation=l['activation'])(flattened)
    output = tf.keras.layers.Dense(1)(flattened)

    model = tf.keras.Model(inputs=[state_in, action_in], outputs=output, name=name)
    model.compile(
        loss=tf.keras.losses.get(config['model_config']['loss']),
        optimizer=tf.keras.optimizers.get(config['model_config']['q_optimizer']),
    )
    return model


def get_policy_model(env, config, name):
    action_shape = env.action_space.sample().shape[0]  # (76, )
    state_shape = env.observation_space.sample().shape[0]  # (76, 2)

    state_in = tf.keras.Input(shape=state_shape)
    flattened = tf.keras.layers.Flatten()(state_in)

    for l in config['model_config']['dense_layers']:
        flattened = tf.keras.layers.Dense(l['size'], activation=l['activation'])(flattened)

    output = tf.keras.layers.Dense(action_shape, activation='tanh')(flattened)

    model = tf.keras.Model(inputs=state_in, outputs=output, name=name)
    model.compile(
        loss=tf.keras.losses.get(config['model_config']['loss']),
        optimizer=tf.keras.optimizers.get(config['model_config']['policy_optimizer']),
    )
    return model


class CAQLModel:
    def __init__(self, env, config):  # purpose can be either 'agent' or 'target'
        self.q_model = get_q_model(env, config, name='q_model')
        self.target_q_model = get_q_model(env, config, name='target_q_model')
        self.policy = get_policy_model(env, config, name='policy')
        self.target_policy = get_policy_model(env, config, name='target_policy')
        self.sync_weights()
        self.config = config
        self.env = env

    def sync_weights(self):
        self.q_model.set_weights(self.target_q_model.get_weights())
        self.policy.set_weights(self.target_policy.get_weights())

    def summary(self, print_fn):
        self.q_model.summary(print_fn=print_fn)
        self.target_q_model.summary(print_fn=print_fn)
        self.policy.summary(print_fn=print_fn)

    @tf.function
    def train(self, states, actions, next_states, rewards, dones):
        # Update Critic
        y = rewards + (1 - dones) * self.config['rl_config']['gamma'] * \
            self.target_q_model([next_states, self.target_policy(next_states, training=True)], training=True)
        critic_value = self.q_model([states, actions], training=True)
        critic_loss = self.q_model.loss(y, critic_value)
        critic_grads = tf.gradients(critic_loss, self.q_model.trainable_variables)
        self.q_model.optimizer.apply_gradients(zip(critic_grads, self.q_model.trainable_variables))

        # Update Actor
        policy_actions = self.policy(states, training=True)
        actor_loss = -tf.reduce_mean(self.q_model([states, policy_actions], training=True))
        actor_grads = tf.gradients(actor_loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(actor_grads, self.policy.trainable_variables))

        # Update Target Networks
        self.update_target_model()

        return dict(
            average_actions=tf.reduce_mean(self.policy(states, training=False), axis=1),
            q=dict(loss=critic_loss, r2=r2(y, critic_value), max_q=tf.reduce_max(y), min_q=tf.reduce_min(y), mean_q=tf.reduce_mean(y)),
            policy=dict(loss=actor_loss, r2=r2(self.target_policy(states, training=False), self.policy(states, training=False)))
        )

    @tf.function
    def update_target_model(self):
        for var, target_var in zip(self.q_model.trainable_variables, self.target_q_model.trainable_variables):
            target_var.assign(
                self.config['rl_config']['tau'] * var + (1 - self.config['rl_config']['tau']) * target_var)

        for var, target_var in zip(self.policy.trainable_variables, self.target_policy.trainable_variables):
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

    def run(self) -> None:
        self.logger.info(f'Initializing Agent {self.index}')

        gpus = tf.config.list_physical_devices('GPU')
        assigned_gpu = self.index % len(gpus)
        tf.config.set_visible_devices(gpus[assigned_gpu], 'GPU')
        tf.config.set_logical_device_configuration(
            gpus[assigned_gpu],
            [tf.config.LogicalDeviceConfiguration(memory_limit=self.config['training_config']['agent_gpu_memory'])])

        self.logger.info(f'Initializing environment.')
        self.env = gym.wrappers.TimeLimit(gym.make('LunarLander-v2', continuous=True), max_episode_steps=2 * self.config['env_config']['time_limit'])
        self.obs = self.env.reset()[0]
        self.done = False

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
        self.noise = getattr(util.rl.exploration, self.config['rl_config']['exploration']['type']) \
            (**self.config['rl_config']['exploration']['config'], shape=self.env.action_space.sample().shape)

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

        for i in range(self.config['env_config']['time_limit']):
            if self.done:
                self.done = False
                self.obs = self.env.reset()[0]

            # if self.epsilon():
            #     action = self.env.action_space.low + np.random.rand(self.env.action_space.shape[0]) * (
            #             self.env.action_space.high - self.env.action_space.low)
            # else:
            action = self.model.target_policy(tf.expand_dims(self.obs, axis=0))[0].numpy()
            action += self.noise()

            next_obs, reward, self.done, truncated, info = self.env.step(action)

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

            if truncated:
                self.obs = self.env.reset()[0]
                break

        # print(f'Experience size: {sys.getsizeof(experiences)} | len: {len(experiences)}')

        return experiences, dict(
            cumulative_reward=rewards,
            average_reward=rewards / count,
            discounted_reward=discounted_rewards,
            episode_length=count,
            noise=self.noise.get_current_noise(),
            time=datetime.now() - start_time,
            # time_report=' ~ '.join([f'{timer}: {time / iterations:.3f}(s/i)' for timer, (time, iterations) in
            #                           visualize.timer_stats.items()])
        )

    def test_trained_model(self):
        start_time = datetime.now()
        env = gym.wrappers.TimeLimit(gym.make('LunarLander-v2', continuous=True), max_episode_steps=self.config['env_config']['time_limit'])
        obs = env.reset()[0]
        done = False
        count = 0
        rewards = 0
        discounted_rewards = 0
        while not done:
            actions = self.model.target_policy(np.expand_dims(obs, axis=0))
            action = actions[0].numpy()

            obs, reward, done, truncated, info = env.step(action)
            count += 1
            rewards += reward
            discounted_rewards += reward * self.config['rl_config']['gamma'] ** count

            if truncated:
                break

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
        while not done:
            actions = self.model.target_policy(np.expand_dims(obs, axis=0))
            action = actions[0].numpy()

            obs, reward, done, truncated, info = self.render_env.step(action)

            if truncated:
                break

    def update_model_weights(self, new_weights):
        self.model.target_policy.set_weights(new_weights)


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

    def update_model(self, samples):
        states = samples['states']
        actions = samples['actions']
        rewards = samples['rewards']
        next_states = samples['next_states']
        dones = samples['dones']

        stats = self.model.train(states, actions, next_states, rewards, dones)

        self.training_step += 1

        return {**dict(
            gamma=self.config['rl_config']['gamma'],
        ), **stats}

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
        self.model.policy.save_weights(path + f'policy_weights-{self.training_step}.h5')
        self.model.target_policy.save_weights(path + f'target_policy_weights-{self.training_step}.h5')

    def load_model(self, run_id, step):
        self.logger.debug(f'Loading model from {run_id}')
        self.model.q_model.load_weights('logs/' + run_id + f'/weights/q_weights-{step}.h5')
        self.model.target_q_model.load_weights('logs/' + run_id + f'/weights/target_q_weights-{step}.h5')
        self.model.policy.load_weights('logs/' + run_id + f'/weights/policy_weights-{step}.h5')
        self.model.target_policy.load_weights('logs/' + run_id + f'/weights/target_policy_weights-{step}.h5')
        return self.model.target_policy.get_weights()

    def run(self) -> None:
        self.init()
        self.logger.info(f'Starting training.')
        while True:
            msg = self.pipe.recv()
            try:
                if msg['type'] == 'update_model':
                    self.logger.debug(f'Updating model.')
                    samples = self.replay_buffer.sample()
                    stats = self.update_model(samples)
                    self.pipe.send(
                        dict(
                            type='model_updated',
                            stats={k: v.numpy() if isinstance(v, tf.Tensor) else v for k, v in stats.items()},
                            weights=self.model.target_policy.get_weights()
                        )
                    )
                elif msg['type'] == 'add_samples':
                    self.logger.debug(f'adding samples.')
                    self.replay_buffer.batch_add(msg['experiences'])
                    self.pipe.send(dict(type='samples_added', total_experience=self.replay_buffer.size()))
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
                    experience_replay_buffer_size = self.trainer_pipe.recv()['total_experience']

                self.log_scalar('rl/total_samples', total_samples, self.training_step)
                self.log_scalar('rl/experience_replay_buffer_size', experience_replay_buffer_size, self.training_step)
                self.log_scalar('rl/noise', np.average([info['noise'] for info in infos]),
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

            with Timer('UpdateModel'):
                for i in range(self.config['training_config']['num_training_per_epoch']):
                    pbar.set_description(
                        f'Updating model {i}/{self.config["training_config"]["num_training_per_epoch"]}...')

                    start = datetime.now()
                    self.trainer_pipe.send(
                        dict(
                            type='update_model'
                        )
                    )
                    ack = self.trainer_pipe.recv()
                    if ack['type'] == 'error':
                        raise Exception(ack['error'])
                    assert ack['type'] == 'model_updated'
                    stats = ack['stats']
                    q_stats = stats['q']
                    policy_stats = stats['policy']
                    average_action_0 = stats['average_actions'][0]
                    average_action_1 = stats['average_actions'][1]

                    new_weights = ack['weights']
                    duration = (datetime.now() - start).total_seconds()

                    self.log_scalar('env/average_action_0', average_action_0, self.training_step)
                    self.log_scalar('env/average_action_1', average_action_1, self.training_step)

                    self.log_scalar('rl/q_loss', q_stats['loss'], self.training_step)
                    self.log_scalar('rl/gamma', stats['gamma'], self.training_step)
                    self.log_scalar('rl/r2', q_stats['r2'], self.training_step)
                    self.log_scalar('rl/max_q', q_stats['max_q'], self.training_step)
                    self.log_scalar('rl/min_q', q_stats['min_q'], self.training_step)
                    self.log_scalar('rl/mean_q', q_stats['mean_q'], self.training_step)
                    self.log_scalar('constants/tau', self.config['rl_config']['tau'], self.training_step)
                    self.log_scalar('training/train_time', duration, self.training_step)
                    # self.log_scalar('rl/policy_loss', policy_stats['loss'], self.training_step)
                    # self.log_scalar('rl/policy_r2', policy_stats['r2'], self.training_step)

                    self.training_step += 1

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

                    self.log_scalar('env/test_reward', np.average([stat['cumulative_reward'] for stat in stats]),
                                    self.training_step)

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
            time_limit=1024
        ),
        model_config=dict(
            q_optimizer=dict(
                class_name='Adam',
                config=dict(
                    learning_rate=1e-3
                )
            ),
            policy_optimizer=dict(
                class_name='Adam',
                config=dict(
                    learning_rate=1e-3
                )
            ),
            loss='MeanSquaredError',
            dense_layers=[
                dict(size=256, activation='relu'),
                dict(size=256, activation='relu'),
            ]
        ),
        rl_config=dict(
            # exploration=dict(
            #     type='NoiseDecay',
            #     config=dict(
            #         noise_start=0.5,
            #         noise_end=0.1,
            #         noise_decay=40_000
            #     )
            exploration=dict(
                type='OUActionNoise',
                config=dict(
                    theta=0.15,
                    mean=0,
                    std_deviation=0.2,
                    dt=0.01,
                    target_scale=0.01,
                    anneal=200_000
                )
            ),
            tau=0.002,
            gamma=0.99,
            batch_size=256,
            buffer_size=500_000,
            num_episodes=10000
        ),
        training_config=dict(
            num_agents=8,
            num_training_per_epoch=32,
            run_id=run_id,
            agent_gpu_memory=512,
            trainer_gpu_memory=512,
            logdir=f'logs/{run_id}',
            checkpoint_interval=20,
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
