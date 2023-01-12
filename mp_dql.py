import logging
import sys
from datetime import datetime
from multiprocessing import Process, Pipe

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm

import util.rl.exploration
from attack_heuristics import Random, Zero
from util.rl.experience_replay import ExperienceReplay
from util.tf.gcn import GraphConvolutionLayer
from util.tf.normalizers import LNormOptimizerLayer
from transport_env.NetworkEnv import TransportationNetworkEnvironment
from transport_env.model import Trip
from util import visualize
from util.visualize import Timer


@tf.function
def get_optimal_action_and_value(actions, states, action_dim, model, action_gradient_step_count, action_optimizer, norm, epsilon):
    # actions.assign(tf.random.normal((states.shape[0], action_dim)))
    # before = model([states, actions])
    # histogram = np.zeros(action_gradient_step_count)
    # action_optimizer = tf.keras.optimizers.Adam(learning_rate=action_optimizer_lr)

    for i in range(action_gradient_step_count):
        # with tf.GradientTape(persistent=True) as tape:
        #     q_value = -tf.reduce_mean(model([states, actions], training=True))

        # grads = tape.gradient(q_value, [actions])
        grads = tf.gradients(-tf.reduce_mean(model([states, actions])), [actions])[0]
        actions.assign_add(grads * action_optimizer.lr)
        normalized, _ = tf.linalg.normalize(actions, axis=1, ord=norm)
        actions.assign(epsilon * normalized)
        # action_optimizer.apply_gradients(zip(grads, actions))
        # actions.assign(
        #     tf.math.divide_no_nan(actions, tf.norm(actions, axis=1, ord=norm, keepdims=True)) * epsilon)

        # histogram[i] = tf.reduce_mean(tf.math.divide_no_nan(model([states, actions]) - before, before)) * 100

    q_values = model([states, actions])

    return actions, q_values


def get_q_model(env, config):
    action_shape = env.action_space.sample().shape  # 76
    state_shape = env.observation_space.sample().shape  # (76, 2)
    adj = env.get_adjacency_matrix()

    state_in = tf.keras.layers.Input(shape=state_shape)
    action_in = tf.keras.layers.Input(shape=action_shape)
    action_reshaped = tf.keras.layers.Reshape((action_shape[0], 1))(action_in)

    shared = tf.keras.layers.Concatenate(axis=2)([state_in, action_reshaped])

    for l in config['model_config']['conv_layers']:
        shared = GraphConvolutionLayer(l['size'], adj, activation=l['activation'])(shared)

    shared = tf.keras.layers.Flatten()(shared)

    for l in config['model_config']['dense_layers']:
        shared = tf.keras.layers.Dense(8, activation='relu')(shared)

    output = tf.keras.layers.Dense(1)(shared)
    model = tf.keras.Model(inputs=[state_in, action_in], outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.get(config['model_config']['optimizer']),
        loss=tf.keras.losses.get(config['model_config']['loss']),
        metrics=[tfa.metrics.RSquare()]
    )

    return model


class Agent(Process):
    def __init__(self, index, config, pipe) -> None:
        super().__init__(name=f'Agent-{index}')
        self.pipe = pipe
        self.config = config
        self.finished = False
        self.index = index

        self.logger = logging.getLogger(f'Agent-{index}')

        self.model = None
        self.env = None
        self.epsilon = None
        self.actions = None
        self.action_optimizer = None

        self.logger.info(f'Agent {self.index} created.')

    def run(self) -> None:
        self.logger.info(f'Initializing Agent {self.index}')

        gpus = tf.config.list_physical_devices('GPU')
        assigned_gpu = self.index % len(gpus)
        tf.config.set_visible_devices(gpus[assigned_gpu], 'GPU')
        tf.config.set_logical_device_configuration(
            gpus[assigned_gpu],
            [tf.config.LogicalDeviceConfiguration(memory_limit=self.config['training_config']['agent_gpu_memory'])])

        # Config contains:
        # - env_config
        # - model_config
        # - rl_config
        # - training_config

        self.logger.info(f'Initializing environment.')
        self.env = TransportationNetworkEnvironment(self.config['env_config'])
        self.logger.info(f'Initializing model.')
        self.model = get_q_model(self.env, self.config)
        self.actions = tf.Variable(tf.random.normal((1, self.env.action_space.shape[0])), name=f'action_agent-{self.index}')
        self.action_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['rl_config']['max_q']['action_optimizer_lr'])
        self.logger.info(f'Initializing exploration strategy.')
        # if self.config['rl_config']['exploration']['type'] == 'constant':
        #     self.epsilon = ConstantEpsilon(self.config['rl_config']['exploration']['config']['value'])
        # elif self.config['rl_config']['exploration']['type'] == 'decaying':
        #     self.epsilon = DecayEpsilon(
        #         self.config['rl_config']['exploration']['config']['initial_epsilon'],
        #         self.config['rl_config']['exploration']['config']['epsilon_end'],
        #         self.config['rl_config']['exploration']['config']['epsilon_decay'])
        # else:
        #     raise ValueError('Unknown epsilon configuration.')
        self.epsilon = getattr(util.rl.exploration, self.config['rl_config']['exploration']['type']) \
            (**self.config['rl_config']['exploration']['config'])

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
            except Exception as e:
                self.logger.exception(e)
                self.pipe.send(dict(type='error', error=str(e)))

    def get_trajectories(self):
        start_time = datetime.now()
        obs = self.env.reset()
        done = False
        count = 0
        rewards = 0
        discounted_rewards = 0
        experiences = []
        # action = self.heuristics[0].predict(obs)
        while not done:
            if self.epsilon():
                action = np.random.choice(self.config['rl_config']['heuristics']).predict(obs)
                action += np.random.normal(0, 1, size=action.shape)
                action /= np.linalg.norm(action, ord=self.config['env_config']['norm'])
            else:
                actions, _ = \
                    get_optimal_action_and_value(
                        self.actions,
                        np.expand_dims(obs, axis=0),
                        self.env.action_space.sample().shape[0],
                        self.model,
                        self.config['rl_config']['max_q']['action_gradient_step_count'],
                        self.action_optimizer,
                        self.config['env_config']['norm'],
                        self.config['env_config']['epsilon']
                    )
                action = actions[0]

            next_obs, reward, done, _ = self.env.step(action)
            next_action = self.config['rl_config']['heuristics'][0].predict(next_obs)

            # Only if it is not the last step due to time limit
            if not done or (self.env.time_step < self.config['env_config']['horizon']):
                experiences.append(dict(
                    state=obs,
                    action=action,
                    reward=reward,
                    next_state=next_obs,
                    next_action=next_action,
                    done=done
                ))
            obs = next_obs
            action = next_action
            count += 1
            rewards += reward
            discounted_rewards += reward * self.config['rl_config']['gamma'] ** count

        return experiences, dict(
            cumulative_reward=rewards,
            average_reward=rewards / count,
            discounted_reward=discounted_rewards,
            episode_length=count,
            epsilon=self.epsilon.get_current_epsilon(),
            time=datetime.now() - start_time
        )

    def update_model_weights(self, new_weights):
        self.model.set_weights(new_weights)
        # for (a, b) in zip(self.model.variables, new_weights):
        #     a.assign(b * config['rl_config']['tau'] + a * (1 - config['rl_config']['tau']))


class Trainer(Process):

    def __init__(self, config, pipe) -> None:
        super().__init__(name='Trainer')
        self.model = None
        self.env = None
        self.logger = logging.getLogger('Trainer')
        self.config = config
        self.pipe = pipe

        # Config contains:
        # - env_config
        # - model_config
        # - rl_config
        # - training_config

        self.replay_buffer = ExperienceReplay(self.config['rl_config']['buffer_size'],
                                              self.config['rl_config']['batch_size'])
        self.training_step = 0

        self.actions = None
        self.action_optimizer = None

    def update_model(self, samples):
        states = samples['states']
        actions = samples['actions']
        rewards = samples['rewards']
        next_states = samples['next_states']
        dones = samples['dones']
        next_actions = samples['next_actions']

        # q_values = self.model([next_states, next_actions])
        _, q_values = get_optimal_action_and_value(
            self.actions,
            next_states,
            self.env.action_space.sample().shape[0],
            self.model,
            self.config['rl_config']['max_q'][
                'action_gradient_step_count'],
            self.action_optimizer,
            self.config['env_config']['norm'],
            self.config['env_config']['epsilon']
        )

        with tf.GradientTape() as tape:
            current_val = self.model([states, actions], training=True)
            target_val = rewards + self.config['rl_config']['gamma'] * q_values * (1 - dones)
            loss = self.model.compiled_loss(target_val, current_val)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        r2 = tfa.metrics.RSquare()
        r2.update_state(target_val, current_val)

        current_lr = self.model.optimizer._decayed_lr(tf.float32).numpy() if callable(
            getattr(self.model.optimizer, '_decayed_lr', None)) else self.model.optimizer.lr.numpy()

        return dict(
            loss=loss.numpy(),
            r2=r2.result().numpy(),
            lr=current_lr,
            gamma=self.config['rl_config']['gamma'],
        )

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
        self.env = TransportationNetworkEnvironment(self.config['env_config'])
        self.logger.info(f'Initializing trainer model.')
        self.model = get_q_model(self.env, self.config)
        self.model.summary()
        self.actions = tf.Variable(tf.random.normal((self.config['rl_config']['batch_size'], self.env.action_space.shape[0])), name='action_optimizer')
        self.action_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['rl_config']['max_q']['action_optimizer_lr'])
        self.logger.info(f'Trainer initialized.')

    def store_model(self):
        path = f'{self.config["training_config"]["logdir"]}/weights-{self.training_step}.h5'
        self.logger.debug(f'Storing model to {path}')
        self.model.save_weights(path)

    def load_model(self, run_id, step):
        self.logger.info(f'Loading model from {self.config["training_config"]["logdir"]}/weights-{step}.h5')
        self.model.load_weights(f'{self.config["training_config"]["logdir"]}/weights-{step}.h5')

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
                            stats=stats,
                            weights=self.model.get_weights()
                        )
                    )
                elif msg['type'] == 'add_samples':
                    self.logger.debug(f'adding samples.')
                    self.replay_buffer.batch_add(msg['experiences'])
                    self.pipe.send(dict(type='samples_added', total_experience=self.replay_buffer.size()))
                elif msg['type'] == 'store_model':
                    self.store_model()
                else:
                    self.logger.error(f'Invalid Message {msg}')
            except Exception as e:
                self.logger.error(f'Error raised {e}')
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
            if msg['type'] == 'log_scalar':
                tf.summary.scalar(msg['name'], msg['value'], msg['step'])
            elif msg['type'] == 'log_histogram':
                tf.summary.histogram(msg['name'], msg['value'], msg['step'])
            else:
                self.logger.error(f'Invalid Message {msg}')


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

        total_samples = 0
        for _ in (pbar := tqdm(range(self.config['rl_config']['num_episodes']))):
            with Timer('GetTrajectories'):
                for agent in self.agents:
                    agent['pipe'].send(dict(
                        type='get_trajectories'
                    ))
                    pbar.set_description(f'Requesting Trajectories')
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
                    new_weights = ack['weights']
                    duration = (datetime.now() - start).total_seconds()

                    self.log_scalar('rl/q_loss', stats['loss'], self.training_step)
                    self.log_scalar('rl/lr', stats['lr'], self.training_step)
                    self.log_scalar('rl/gamma', stats['gamma'], self.training_step)
                    self.log_scalar('rl/r2', stats['r2'], self.training_step)
                    self.log_scalar('training/train_time', duration, self.training_step)

                    self.training_step += 1

                pbar.set_description(f'Sending updated weights.')

                for agent in self.agents:
                    agent['pipe'].send(dict(
                        type='update_weights',
                        weights=new_weights
                    ))

                for agent in self.agents:
                    assert agent['pipe'].recv()['type'] == 'weights_updated'

                if self.training_step % self.config['training_config']['checkpoint_interval'] == 0:
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
            city='SiouxFalls',
            horizon=50,
            epsilon=30,
            norm=2,
            frac=0.25,
            num_sample=20,
            render_mode=None,
            reward_multiplier=1.0,
            congestion=True,
            trips=dict(
                type='demand_file',
                trips=Trip.trips_using_demand_file('Sirui/traffic_data/sf_demand.txt'),
                strategy='random',
                count=100
            ),
            rewarding_rule='vehicle_count',
        ),
        model_config=dict(
            optimizer=dict(
                class_name='Adam',
                config=dict(
                    # learning_rate=1e-5
                    learning_rate=dict(
                        class_name='ExponentialDecay',
                        config=dict(
                            initial_learning_rate=1e-4,
                            decay_steps=400,
                            decay_rate=0.95
                        )
                    )
                )
            ),
            loss='mse',
            conv_layers=[
                dict(size=64, activation='elu'),
                dict(size=64, activation='elu'),
                dict(size=64, activation='elu'),
                dict(size=64, activation='elu'),
                dict(size=64, activation='elu'),
                dict(size=64, activation='elu'),
                dict(size=64, activation='elu'),
            ],
            dense_layers=[
                dict(size=128, activation='elu'),
                dict(size=128, activation='elu'),
                dict(size=128, activation='elu'),
            ]
        ),
        rl_config=dict(
            exploration=dict(
                # type='ConstantEpsilon',
                # config=dict(
                #     value=0.2
                # ),
                type='DecayEpsilon',
                config=dict(
                    epsilon_start=0.5,
                    epsilon_end=0.01,
                    epsilon_decay=3000
                )
            ),
            heuristics=list(),
            max_q=dict(
                action_gradient_step_count=100,
                action_optimizer_lr=1e-3,
            ),
            gamma=0.95,
            batch_size=512,
            buffer_size=50000,
            num_episodes=1000
        ),
        training_config=dict(
            num_agents=35,
            num_training_per_epoch=8,
            run_id=run_id,
            agent_gpu_memory=512,
            trainer_gpu_memory=512,
            logdir=f'logs/{run_id}',
            checkpoint_interval=10
        ),
    )

    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s',
                        level=logging.INFO
                        # level=0
                        )

    logger = logging.getLogger(__name__)
    logger.info(f'Starting run {run_id}...')

    config['rl_config']['heuristics'] = [
        # GreedyRiderVector(config['env_config']['epsilon'], config['env_config']['norm']),
        Random((76,), config['env_config']['norm'], config['env_config']['epsilon'], config['env_config']['frac'],
               'discrete'),
        Zero((76,)),
    ]

    manager = Manager(config)

    try:
        manager.train()
    except BaseException as e:
        logger.exception(f'Exception {e}')
    manager.killall()
