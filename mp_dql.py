import logging
import sys
from datetime import datetime
from multiprocessing import Process, Pipe

import numpy as np
import tensorflow_addons as tfa
from tqdm import tqdm

from attack_heuristics import GreedyRiderVector, Random, Zero
from rl_util.experience_replay import ExperienceReplay
from rl_util.exploration import ConstantEpsilon
from tf_util.gcn import GraphConvolutionLayer
from transport_env.NetworkEnv import TransportationNetworkEnvironment
from transport_env.model import Trip
from util import visualize
from util.visualize import Timer


def get_optimal_action_and_value(states, action_dim, model, action_gradient_step_count, action_optimizer_lr, norm,
                                 epsilon):
    import tensorflow as tf
    actions = tf.Variable(tf.random.normal((states.shape[0], action_dim)), name='action')
    before = model([states, actions])
    histogram = np.zeros(action_gradient_step_count)
    action_optimizer = tf.keras.optimizers.Adam(learning_rate=action_optimizer_lr)

    for i in range(action_gradient_step_count):
        with tf.GradientTape(persistent=True) as tape:
            q_value = -tf.reduce_mean(model([states, actions], training=True))

        grads = tape.gradient(q_value, [actions])
        action_optimizer.apply_gradients(zip(grads, [actions]))
        actions.assign(
            tf.math.divide_no_nan(actions, tf.norm(actions, axis=1, ord=norm, keepdims=True)) * epsilon)

        histogram[i] = tf.reduce_mean(tf.math.divide_no_nan(model([states, actions]) - before, before)) * 100

    q_values = model([states, actions])

    return actions, q_values, tf.convert_to_tensor(histogram)


def get_q_model(env, config):
    import tensorflow as tf
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
        optimizer=config['model_config']['optimizer']['type'],
        loss=config['model_config']['loss'],
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

        self.logger.info(f'Agent {self.index} created.')

    def run(self) -> None:
        self.logger.info(f'Initializing Agent {self.index}')

        import tensorflow as tf

        gpus = tf.config.list_physical_devices('GPU')
        tf.config.set_logical_device_configuration(
            gpus[self.index % len(gpus)],
            [tf.config.LogicalDeviceConfiguration(memory_limit=512)])

        # Config contains:
        # - env_config
        # - model_config
        # - rl_config
        # - training_config

        self.logger.info(f'Initializing environment.')
        self.env = TransportationNetworkEnvironment(self.config['env_config'])
        self.logger.info(f'Initializing model.')
        self.model = get_q_model(self.env, self.config)
        self.logger.info(f'Initializing exploration strategy.')
        if self.config['rl_config']['exploration']['type'] == 'constant':
            self.epsilon = ConstantEpsilon(self.config['rl_config']['exploration']['epsilon'])
        else:
            raise ValueError('Unknown epsilon configuration.')

        self.logger.info(f'Agent {self.index} started.')
        while not self.finished:
            message = self.pipe.recv()
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
            else:
                action = \
                    get_optimal_action_and_value(
                        np.expand_dims(obs, axis=0),
                        self.env.action_space.sample().shape[0],
                        self.model,
                        self.config['rl_config']['max_q']['action_gradient_step_count'],
                        self.config['rl_config']['max_q']['action_optimizer_lr'],
                        self.config['env_config']['norm'],
                        self.config['env_config']['epsilon']
                    )[0][0]

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
            time=datetime.now() - start_time
        )

    def update_model_weights(self, new_weights):
        self.model.set_weights(new_weights)


class Trainer:

    def __init__(self, config) -> None:
        super().__init__()
        self.logger = logging.getLogger('Trainer')
        self.config = config

        self.logger.info(f'Initializing trainer.')
        self.logger.info(f'Initializing trainer environment variables.')
        self.env = TransportationNetworkEnvironment(config['env_config'])
        self.logger.info(f'Initializing trainer model.')
        self.model = get_q_model(self.env, config)
        self.logger.info(f'Trainer initialized.')

        # Config contains:
        # - env_config
        # - model_config
        # - rl_config
        # - training_config

        self.agents = []
        for i in range(config['training_config']['num_cpu']):
            self.logger.info(f'Creating agent {i}.')
            parent_conn, child_conn = Pipe()
            agent = Agent(i, config, child_conn)
            self.agents.append(
                dict(
                    agent=agent,
                    pipe=parent_conn
                )
            )
            agent.start()

        self.replay_buffer = ExperienceReplay(self.config['rl_config']['buffer_size'],
                                              self.config['rl_config']['batch_size'])
        self.training_step = 0

    def update_model(self, samples):
        import tensorflow as tf
        states = samples['states']
        actions = samples['actions']
        rewards = samples['rewards']
        next_states = samples['next_states']
        dones = samples['dones']
        next_actions = samples['next_actions']

        # q_values = self.model([next_states, next_actions])
        _, q_values, histogram = get_optimal_action_and_value(
            next_states,
            self.env.action_space.sample().shape[0],
            self.model,
            self.config['rl_config']['max_q'][
                'action_gradient_step_count'],
            self.config['rl_config']['max_q'][
                'action_optimizer_lr'],
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

        tf.summary.scalar('rl/q_loss', data=loss, step=self.training_step)
        tf.summary.scalar('rl/lr', data=self.model.optimizer._decayed_lr(tf.float32), step=self.training_step)
        tf.summary.scalar('rl/r2', data=r2.result(), step=self.training_step)

        tf.summary.scalar('rl/gamma',
                          data=self.config['rl_config']['gamma'],
                          step=self.training_step)

    def train(self):
        import tensorflow as tf
        total_samples = 0
        for _ in (pbar := tqdm(range(self.config['rl_config']['num_episodes']))):
            with Timer('GetTrajectories'):
                for agent in self.agents:
                    agent['pipe'].send(dict(
                        type='get_trajectories'
                    ))
                infos = []
                for i, agent in enumerate(self.agents):
                    msg = agent['pipe'].recv()
                    assert msg['type'] == 'trajectories'
                    self.replay_buffer.batch_add(msg['experiences'])
                    infos.append(msg['info'])
                    total_samples += msg['info']['episode_length']
                    pbar.set_description(f'Received {i + 1}/{len(self.agents)} trajectories')

            if self.replay_buffer.size() > self.config['rl_config']['batch_size']:
                with Timer('UpdateModel'):
                    for _ in range(self.config['training_config']['num_training_per_epoch']):
                        self.update_model(self.replay_buffer.sample())
                        self.training_step += 1

                    for agent in self.agents:
                        agent['pipe'].send(dict(
                            type='update_weights',
                            weights=self.model.get_weights()
                        ))
                        assert agent['pipe'].recv()['type'] == 'weights_updated'

            tf.summary.scalar('rl/total_samples', data=total_samples, step=self.training_step)
            tf.summary.histogram('rl/episode_length', data=[info['episode_length'] for info in infos], step=self.training_step)
            tf.summary.histogram('rl/cumulative_reward', data=[info['cumulative_reward'] for info in infos], step=self.training_step)
            tf.summary.histogram('rl/average_reward', data=[info['average_reward'] for info in infos], step=self.training_step)
            tf.summary.histogram('rl/discounted_reward', data=[info['discounted_reward'] for info in infos], step=self.training_step)
            tf.summary.histogram('rl/time', data=[info['time'].total_seconds() for info in infos], step=self.training_step)

            time_report = ' ~ '.join([f'{timer}: {time / iterations:.3f}(s/i)' for timer, (time, iterations) in
                                      visualize.timer_stats.items()])
            # pbar.set_description(f'{time_report}')

    def store_model(self):
        path = f'{self.config["training_config"]["logdir"]}/model/weights'
        self.logger.info(f'Storing model to {path}')
        self.model.save_weights(path)

    def load_model(self, run_id):
        self.logger.info(f'Loading model from logs/{run_id}/model/weights')
        self.model.load_weights(f'logs/{run_id}/model/weights')

    def killall(self):
        for agent in self.agents:
            agent['agent'].kill()


if __name__ == '__main__':
    import tensorflow as tf
    config = dict(
        env_config=dict(
            city='SiouxFalls',
            horizon=50,
            epsilon=11,
            norm=1,
            frac=0.5,
            num_sample=20,
            render_mode=None,
            reward_multiplier=1.0,
            congestion=True,
            trips=dict(
                type='demand_file',
                trips=Trip.trips_using_demand_file('Sirui/traffic_data/sf_demand.txt'),
                strategy='random',
                count=100
            )
        ),
        model_config=dict(
            optimizer=dict(
                type='adam',
                lr=1e-4,
            ),
            loss='mse',
            conv_layers=[
                dict(size=8, activation='elu'),
                dict(size=8, activation='elu'),
                dict(size=8, activation='elu'),
                dict(size=8, activation='elu'),
            ],
            dense_layers=[
                dict(size=8, activation='relu'),
            ]
        ),
        rl_config=dict(
            exploration=dict(
                type='constant',
                epsilon=0.1,
            ),
            heuristics=list(),
            max_q=dict(
                action_gradient_step_count=7,
                action_optimizer_lr=1e-3,
            ),
            gamma=0.95,
            batch_size=64,
            buffer_size=5000,
            num_episodes=10000
        ),
        training_config=dict(
            num_cpu=32,
            num_training_per_epoch=1
        ),
    )

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

    config['rl_config']['heuristics'] = [
        GreedyRiderVector(config['env_config']['epsilon'], config['env_config']['norm']),
        Random((76, ), config['env_config']['norm'], config['env_config']['epsilon'], config['env_config']['frac'], 'discrete'),
        Zero((76, )),
    ]

    trainer = Trainer(config)
    try:
        trainer.train()
    except KeyboardInterrupt:
        trainer.killall()

