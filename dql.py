import logging
import random
import sys
import math

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from attack_heuristics import BaseHeuristic, Random
from transport_env.NetworkEnv import TransportationNetworkEnvironment
from transport_env.model import Trip


def update_model(model, loss_function, optimizer, samples, gamma, action_gradient_step, action_optimizer):
    states = samples['states']
    actions = samples['actions']
    rewards = samples['rewards']
    next_states = samples['next_states']
    dones = samples['dones']
    _, q_values = get_optimal_action_and_value(model, next_states, actions.shape[1], action_optimizer, action_gradient_step)

    with tf.GradientTape() as tape:
        current_val = model([states, actions], training=True)
        target_val = rewards + gamma * q_values * (1 - dones)
        loss = tf.reduce_mean(loss_function(target_val, current_val))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


# def get_optimal_action_and_value(model, states, action_dim, action_optimizer, action_gradient_step_count):
#     actions = tf.Variable(tf.random.normal((states.shape[0], action_dim)), name='action')
#
#     with tf.GradientTape(persistent=True) as tape:
#         for _ in range(action_gradient_step_count):
#             q_values = -tf.reduce_mean(model([states, actions], training=True))
#
#         grads = tape.gradient(q_values, [actions])
#         action_optimizer.apply_gradients(zip(grads, [actions]))
#
#     q_values = model([states, actions])
#
#     return actions, q_values

def get_optimal_action_and_value(model, states, action_dim, action_optimizer, action_gradient_step_count):
    actions = tf.Variable(tf.random.normal((states.shape[0], action_dim)), name='action')

    for _ in range(action_gradient_step_count):
        action_optimizer.minimize(lambda: -tf.reduce_mean(model([states, actions])), var_list=[actions])

    q_values = model([states, actions])

    return actions, q_values


def get_q_model(env):
    action_dim = env.action_space.sample().shape[0]
    state_dim = np.prod(env.observation_space.shape)

    state_in = tf.keras.layers.Input(shape=(state_dim,))
    state_out = tf.keras.layers.Dense(64, activation='relu')(state_in)
    state_out = tf.keras.layers.Dense(64, activation='relu')(state_out)
    state_out = tf.keras.layers.Dense(64, activation='relu')(state_out)

    action_in = tf.keras.layers.Input(shape=(action_dim,))
    action_out = tf.keras.layers.Dense(16, activation='relu')(action_in)
    action_out = tf.keras.layers.Dense(16, activation='relu')(action_out)

    concat = tf.keras.layers.Concatenate()([state_out, action_out])

    shared = tf.keras.layers.Dense(128, activation='relu')(concat)
    shared = tf.keras.layers.Dense(128, activation='relu')(shared)

    output = tf.keras.layers.Dense(1)(shared)
    model = tf.keras.Model(inputs=[state_in, action_in], outputs=output)
    return model


class ExperienceReplay:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = []

    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append(
            dict(
                state=obs,
                action=action,
                reward=reward,
                next_state=next_obs,
                done=done,
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
        )

        experiences = random.choices(self.buffer, k=self.batch_size)

        for e in experiences:
            ret['states'].append(e['state'].flatten())
            ret['actions'].append(e['action'])
            ret['rewards'].append(e['reward'])
            ret['next_states'].append(e['next_state'].flatten())
            ret['dones'].append(e['done'])

        return {k: tf.convert_to_tensor(v, dtype=tf.float32) for k, v in ret.items()}


def get_trajectories(env, model, replay_buffer, action_optimizer, action_gradient_step_count, epsilon_function, heuristic: BaseHeuristic):
    obs = env.reset()
    done = False
    count = 0
    rewards = 0
    while not done:
        if epsilon_function():
            # TODO the following line does not work with GreedyRider
            action = heuristic.predict(obs)
        else:
            action = get_optimal_action_and_value(model, np.expand_dims(obs.flatten(), axis=0), env.action_space.shape[0], action_optimizer, action_gradient_step_count)[0][0]
            action = tf.nn.relu(action)
            action = action / tf.norm(action, ord=env.config['norm']) * env.config['epsilon']

        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs
        count += 1
        rewards += reward

    return count, rewards/count


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
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.step / self.epsilon_decay)
        self.step += 1
        return random.random() < epsilon

    def get_current_epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.step / self.epsilon_decay)


if __name__ == '__main__':

    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    logger = logging.getLogger(__name__)

    config = dict(
        city='SiouxFalls',
        horizon=60,
        epsilon=10,  # range(1, 11, 2)
        norm=np.inf,
        frac=0.5,
        num_sample=20,
        repeat=1000,
        render_mode=None,
        reward_multiplier=0.01
    )

    rl_config = dict(
        buffer_size=10000,
        batch_size=128,
        # heuristic_probability=0.1,
        decay_start=1,
        decay_end=0.01,
        decay_steps=10000,
        num_episodes=10000,
        gamma=0.95,
        learning_rate=0.001,
        action_gradient_step=0.01,
        action_gradient_step_count=20,
        expected_update_per_trajectory=1
    )

    logger.info(f'Config: {config}')
    logger.info(f'RL Config: {rl_config}')

    config.update(dict(trip_generator=Trip.using_demand_file(f'Sirui/traffic_data/sf_demand.txt', 'top', 20)))
    env = TransportationNetworkEnvironment(config)
    model = get_q_model(env)
    # TODO GreedyRider does not work due to observation mismatch.
    # heuristic = GreedyRider(env.action_space, config['epsilon'], config['norm'])
    epsilon_decay = DecayEpsilon(rl_config['decay_start'], rl_config['decay_end'], rl_config['decay_steps'])
    heuristic = Random(env.action_space, config['norm'], config['epsilon'], config['frac'], 'discrete')
    replay_buffer = ExperienceReplay(rl_config['buffer_size'], rl_config['batch_size'])
    loss_function = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=rl_config['learning_rate'])
    action_optimizer = tf.keras.optimizers.Adam(learning_rate=rl_config['action_gradient_step'])

    for training_step in (pbar := tqdm(range(rl_config['num_episodes']))):
        length, reward = get_trajectories(env, model, replay_buffer, action_optimizer, rl_config['action_gradient_step_count'], epsilon_decay, heuristic)
        pbar.set_description(f'Length {length} - Reward {reward * length:.2f} - Buffer Size {replay_buffer.size()} - Epsilon {epsilon_decay.get_current_epsilon():.2f}')
        if replay_buffer.size() > rl_config['batch_size']:
            for i in range(length * rl_config['expected_update_per_trajectory']):
                update_model(model, loss_function, optimizer, replay_buffer.sample(), rl_config['gamma'], rl_config['action_gradient_step_count'], action_optimizer)

