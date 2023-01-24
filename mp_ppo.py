import logging
import sys
from datetime import datetime
from multiprocessing import Process, Pipe

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from tqdm import tqdm

from attack_heuristics import Random, Zero, GreedyRiderVector
from transport_env.NetworkEnv import TransportationNetworkEnvironment
from transport_env.model import Trip
from util import visualize
from util.tf.gcn import GraphConvolutionLayer
from util.visualize import Timer


def calculate_gae(rewards, state_val, next_state_val, dones, gamma, lam):
    gae = 0
    gae_list = tf.TensorArray(rewards.dtype, size=rewards.shape[0])
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_state_val[i][0] * (1 - dones[i]) - state_val[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        gae_list = gae_list.write(i, gae)
    return gae_list.stack()


def get_vf(env, config):
    # action_shape = env.action_space.sample().shape  # 76
    state_shape = env.observation_space.sample().shape  # (76, 2)
    adj = env.get_adjacency_matrix()

    input = tf.keras.layers.Input(shape=state_shape)
    # action_in = tf.keras.layers.Input(shape=action_shape)
    # action_reshaped = tf.keras.layers.Reshape((action_shape[0], 1))(action_in)

    # shared = tf.keras.layers.Concatenate(axis=2)([state_in, action_reshaped])
    shared = input

    for l in config['model_config']['conv_layers']:
        shared = GraphConvolutionLayer(l['size'], adj, activation=l['activation'])(shared)

    shared = tf.keras.layers.Flatten()(shared)

    for l in config['model_config']['dense_layers']:
        shared = tf.keras.layers.Dense(8, activation=l['activation'])(shared)

    output = tf.keras.layers.Dense(1)(shared)
    model = tf.keras.Model(inputs=input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.get(config['model_config']['optimizer']),
    )

    return model


def get_policy(env, config):
    action_shape = env.action_space.sample().shape  # (76, )
    state_shape = env.observation_space.sample().shape  # (76, 2)
    adj = env.get_adjacency_matrix()

    input = tf.keras.layers.Input(shape=state_shape)
    # action_in = tf.keras.layers.Input(shape=action_shape)
    # action_reshaped = tf.keras.layers.Reshape((action_shape[0], 1))(action_in)

    # shared = tf.keras.layers.Concatenate(axis=2)([state_in, action_reshaped])
    shared = input

    for l in config['model_config']['conv_layers']:
        shared = GraphConvolutionLayer(l['size'], adj, activation=l['activation'])(shared)

    shared = tf.keras.layers.Flatten()(shared)

    for l in config['model_config']['dense_layers']:
        shared = tf.keras.layers.Dense(8, activation=l['activation'])(shared)

    output = tf.keras.layers.Dense(action_shape[0] * 2)(shared)
    model = tf.keras.Model(inputs=input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.get(config['model_config']['optimizer'])
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

        self.logger.info(f'Agent {self.index} created.')

    def run(self) -> None:
        self.logger.info(f'Initializing Agent {self.index}')

        gpus = tf.config.list_physical_devices('GPU')
        assigned_gpu = self.index % len(gpus)
        tf.config.set_visible_devices(gpus[assigned_gpu], 'GPU')
        tf.config.set_logical_device_configuration(
            gpus[assigned_gpu],
            [tf.config.LogicalDeviceConfiguration(memory_limit=self.config['training_config']['agent_gpu_memory'])])

        self.logger.info(f'Initializing environment.')
        self.env = TransportationNetworkEnvironment(self.config['env_config'])
        self.logger.info(f'Initializing model.')
        self.policy = get_policy(self.env, self.config)

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
        while not done:
            action_logits = self.policy(tf.expand_dims(obs, axis=0))
            assert not np.isnan(action_logits).any(), f'Action_Logits has nan: {action_logits}'

            means, stds = tf.split(action_logits, 2, axis=1)
            distribution = tfp.distributions.MultivariateNormalDiag(loc=means, scale_diag=stds)
            action = distribution.sample()
            unscaled_action = action.numpy()[0]
            action = tf.nn.relu(unscaled_action).numpy()
            norm = np.linalg.norm(action, ord=self.config['env_config']['norm'])
            action = self.config['env_config']['epsilon'] * np.divide(action, norm, where=norm != 0)

            obs, reward, done, _ = self.env.step(action)

            # Only if it is not the last step due to time limit
            if not done or (self.env.time_step < self.config['env_config']['horizon']):
                experiences.append(dict(
                    state=obs,
                    action=unscaled_action,
                    reward=reward,
                    next_state=obs,
                    done=done
                ))
            count += 1
            rewards += reward
            discounted_rewards += reward * self.config['rl_config']['gamma'] ** count

        return experiences, dict(
            cumulative_reward=rewards,
            average_reward=rewards / count,
            discounted_reward=discounted_rewards,
            episode_length=count,
            time=datetime.now() - start_time
        )

    def test_trained_model(self):
        start_time = datetime.now()
        obs = self.env.reset()
        done = False
        count = 0
        rewards = 0
        discounted_rewards = 0
        while not done:
            action_logits = self.policy(tf.expand_dims(obs, axis=0))
            assert not np.isnan(action_logits).any(), f'Action_Logits has nan: {action_logits}'

            means, stds = tf.split(action_logits, 2, axis=1)
            distribution = tfp.distributions.MultivariateNormalDiag(loc=means, scale_diag=tf.zeros(stds.shape))
            action = distribution.sample()
            action = tf.nn.relu(action)
            action = tf.nn.relu(action).numpy()[0]
            norm = np.linalg.norm(action, ord=self.config['env_config']['norm'])
            action = self.config['env_config']['epsilon'] * np.divide(action, norm, where=norm != 0)

            obs, reward, done, _ = self.env.step(action)
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

    def update_model_weights(self, new_weights):
        self.policy.set_weights(new_weights)


class Trainer(Process):

    def __init__(self, config, pipe) -> None:
        super().__init__(name='Trainer')
        self.env = None
        self.logger = logging.getLogger('Trainer')
        self.config = config
        self.pipe = pipe

        # Config contains:
        # - env_config
        # - model_config
        # - rl_config
        # - training_config

        self.policy = None
        self.value_function = None

        self.beta = None

        self.training_step = 0

    def update_model(self, samples):
        states = np.array([sample['state'] for sample in samples], dtype=np.float32)
        actions = np.array([sample['action'] for sample in samples], dtype=np.float32)
        rewards = np.array([sample['reward'] for sample in samples], dtype=np.float32)
        next_states = np.array([sample['next_state'] for sample in samples], dtype=np.float32)
        dones = np.array([sample['done'] for sample in samples], dtype=np.float32)

        old_action_logits = self.policy(states)
        old_means, old_stds = tf.split(old_action_logits, 2, axis=1)
        old_distributions = tfp.distributions.MultivariateNormalDiag(loc=old_means, scale_diag=old_stds)
        old_log_probs = old_distributions.log_prob(actions)

        next_state_val = self.value_function(next_states)

        with tf.GradientTape(persistent=True) as tape:
            state_val = self.value_function(states)
            action_logits = self.policy(states)
            means, stds = tf.split(action_logits, 2, axis=1)
            distribution = tfp.distributions.MultivariateNormalDiag(loc=means, scale_diag=stds)
            log_probs = distribution.log_prob(actions)

            kl = old_distributions.kl_divergence(distribution)
            r_t = tf.math.exp(log_probs - old_log_probs)
            gae = calculate_gae(rewards, state_val, next_state_val, dones, self.config['rl_config']['gamma'],
                                self.config['rl_config']['lam'])
            entropy = distribution.entropy()

            vf_target = tf.expand_dims(rewards, axis=0) + self.config['rl_config']['gamma'] * next_state_val * (
                        1 - tf.expand_dims(dones, axis=0))

            l_vf = tf.reduce_mean(
                tf.square(vf_target - state_val))
            l_clip = - tf.reduce_mean(
                tf.math.minimum(tf.clip_by_value(r_t, 1 - self.config['rl_config']['epsilon'],
                                                 1 + self.config['rl_config']['epsilon']) * gae, r_t * gae))
            l_kl = tf.reduce_mean(kl)
            l_entropy = - tf.reduce_mean(entropy)

            surrogate = self.beta * l_kl + l_clip + self.config['rl_config']['c2'] * l_entropy + \
                        self.config['rl_config']['c1'] * l_vf

        policy_grads = tape.gradient(surrogate, self.policy.trainable_variables)
        value_grads = tape.gradient(surrogate, self.value_function.trainable_variables)

        self.policy.optimizer.apply_gradients(zip(policy_grads, self.policy.trainable_variables))
        self.value_function.optimizer.apply_gradients(zip(value_grads, self.value_function.trainable_variables))

        if l_kl < self.config['rl_config']['target_beta'] / 1.5:
            self.beta /= 2
        elif l_kl > self.config['rl_config']['target_beta'] * 1.5:
            self.beta *= 2

        r2 = tfa.metrics.RSquare()
        r2.update_state(vf_target, state_val)

        self.training_step += 1

        policy_lr = self.policy.optimizer._decayed_lr(tf.float32).numpy() if callable(
            getattr(self.policy.optimizer, '_decayed_lr', None)) else self.policy.optimizer.lr.numpy()

        vf_lr = self.value_function.optimizer._decayed_lr(tf.float32).numpy() if callable(
            getattr(self.value_function.optimizer, '_decayed_lr', None)) else self.value_function.optimizer.lr.numpy()

        return dict(
            vf_loss=l_vf.numpy(),
            gae=tf.reduce_mean(gae).numpy(),
            entropy=l_entropy.numpy(),
            vf_r2=r2.result().numpy(),
            vf_lr=policy_lr,
            policy_lr=vf_lr,
            epsilon=self.config['rl_config']['epsilon'],
            gamma=self.config['rl_config']['gamma'],
            lam=self.config['rl_config']['lam'],
            c1=self.config['rl_config']['c1'],
            c2=self.config['rl_config']['c2'],
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
        self.policy = get_policy(self.env, self.config)
        self.value_function = get_vf(self.env, self.config)
        self.policy.summary()
        self.value_function.summary()
        self.logger.info(f'Trainer initialized.')

        self.beta = self.config['rl_config']['initial_beta']

    def store_model(self):
        policy_path = f'logs/{self.config["training_config"]["run_id"]}/policy-{self.training_step}.h5'
        vf_path = f'logs/{self.config["training_config"]["run_id"]}/vf-{self.training_step}.h5'
        self.logger.debug(f'Storing model to {policy_path} and {vf_path}')
        self.policy.save_weights(policy_path)
        self.value_function.save_weights(vf_path)

    def load_model(self, run_id, step):
        self.logger.info(f'Loading model from logs/{run_id}/vf-{step}.h5 and logs/{run_id}/policy-{step}.h5')
        self.policy.load_weights(f'logs/{run_id}/policy-{step}.h5')
        self.value_function.load_weights(f'logs/{run_id}/vf-{step}.h5')

    def run(self) -> None:
        self.init()
        self.logger.info(f'Starting training.')
        while True:
            msg = self.pipe.recv()
            try:
                if msg['type'] == 'update_model':
                    self.logger.debug(f'Updating model.')
                    stats = self.update_model(msg['experiences'])
                    self.pipe.send(
                        dict(
                            type='model_updated',
                            stats=stats,
                            weights=self.policy.get_weights()
                        )
                    )
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
        for trajectory_step in (pbar := tqdm(range(self.config['rl_config']['num_episodes']))):
            with Timer('GetTrajectories'):
                for i, agent in enumerate(self.agents):
                    agent['pipe'].send(dict(
                        type='get_trajectories'
                    ))
                    pbar.set_description(f'Requesting Trajectories {i}/{len(self.agents)}')
                infos = []
                experiences = []
                for i, agent in enumerate(self.agents):
                    pbar.set_description(f'Received {i}/{len(self.agents)} trajectories')
                    msg = agent['pipe'].recv()
                    if msg['type'] == 'error':
                        raise Exception(msg['error'])
                    experiences.append(msg['experiences'])
                    infos.append(msg['info'])
                    total_samples += msg['info']['episode_length']

                self.log_scalar('rl/total_samples', total_samples, self.training_step)
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
                for i, experience in enumerate(experiences):
                    pbar.set_description(
                        f'Updating model {i}/{len(experiences)}...')

                    start = datetime.now()
                    self.trainer_pipe.send(
                        dict(
                            type='update_model',
                            experiences=experience,
                        )
                    )
                    ack = self.trainer_pipe.recv()
                    if ack['type'] == 'error':
                        raise Exception(ack['error'])
                    assert ack['type'] == 'model_updated'
                    stats = ack['stats']
                    new_weights = ack['weights']
                    duration = (datetime.now() - start).total_seconds()

                    self.log_scalar('rl/vf_loss', stats['vf_loss'], self.training_step)
                    self.log_scalar('rl/gae', stats['gae'], self.training_step)
                    self.log_scalar('rl/entropy', stats['entropy'], self.training_step)
                    self.log_scalar('rl/vf_r2', stats['vf_r2'], self.training_step)
                    self.log_scalar('rl/vf_lr', stats['vf_lr'], self.training_step)
                    self.log_scalar('rl/policy_lr', stats['policy_lr'], self.training_step)

                    self.log_scalar('constants/gamma', stats['gamma'], self.training_step)
                    self.log_scalar('constants/lam', stats['lam'], self.training_step)
                    self.log_scalar('constants/c1', stats['c1'], self.training_step)
                    self.log_scalar('constants/c2', stats['c2'], self.training_step)
                    self.log_scalar('constants/epsilon', stats['epsilon'], self.training_step)

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

                if trajectory_step % self.config['training_config']['checkpoint_interval'] == 0:
                    self.logger.debug(f'Checkpointing')

                    for agent in self.agents:
                        agent['pipe'].send(dict(
                            type='test'
                        ))

                    stats = []
                    for i, agent in enumerate(self.agents):
                        pbar.set_description(f'Testing {i}/{len(self.agents)}.')
                        msg = agent['pipe'].recv()
                        if msg['type'] == 'error':
                            raise Exception(msg['error'])
                        stats.append(msg['stat'])

                    self.log_scalar('env/test_reward', np.average([stat['cumulative_reward'] for stat in stats]),
                                    self.training_step)

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
            frac=0.1,
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
                dict(size=64, activation='elu'),
                dict(size=64, activation='tanh'),
            ]
        ),
        rl_config=dict(
            gamma=0.95,
            lam=1.0,
            epsilon=0.3,
            c1=0.1,
            c2=0.1,
            initial_beta=0.1,
            target_beta=0.5,
            num_episodes=1000
        ),
        training_config=dict(
            num_agents=1,
            num_training_per_epoch=4,
            run_id=run_id,
            agent_gpu_memory=512,
            trainer_gpu_memory=512,
            logdir=f'logs/{run_id}',
            checkpoint_interval=5
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
        GreedyRiderVector(config['env_config']['epsilon'], config['env_config']['norm']),
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
