import logging
import sys
import warnings

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from util.tf.gcn import GraphConvolutionLayer
from transport_env.NetworkEnv import TransportationNetworkEnvironment
from transport_env.model import Trip


def get_reward_from_state(env, state):
    r = 0
    for i, e in enumerate(env.base.edges):
        r += (state[i][0] + state[i][3] + state[i][4]) * env.get_travel_time(*e, 0)
    return r


def r2(y, y_pred):
    return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)


@tf.keras.utils.register_keras_serializable(package='Custom', name='l0')
class L0Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, l0=0.01):
        self.l0 = l0

    def __call__(self, x):
        return tf.cast(tf.math.count_nonzero(x), tf.float64) * self.l0

    def get_config(self):
        return {'l0': float(self.l0)}


if __name__ == '__main__':

    logging.basicConfig(
        stream=sys.stdout,
        format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s',
        level=logging.INFO
        # level=0
    )
    warnings.filterwarnings('ignore')

    config = dict(
        city='SiouxFalls',
        horizon=100,
        epsilon=3,
        norm=2,
        frac=0.5,
        num_sample=20,
        render_mode=None,
        reward_multiplier=1,
        congestion=False,
    )

    rl_config = dict(
        buffer_size=20_000,
        batch_size=256,
        # decay_start=1,
        # decay_end=0.01,
        # decay_steps=10000,
        epsilon=0.0,
        num_episodes=100000,
        gamma=1.0,
        learning_rate=0.01,
        action_gradient_step=0.001,
        action_gradient_step_count=20,
        num_training_per_epoch=1,
    )

    config.update(dict(trip_generator=Trip.using_demand_file(f'Sirui/traffic_data/sf_demand.txt', 'random', 32)))
    env = TransportationNetworkEnvironment(config)

    adj = env.get_adjacency_matrix()
    state_in = tf.keras.layers.Input(shape=env.observation_space.sample().shape)

    shared = GraphConvolutionLayer(5, adj, activation='elu')(state_in)
    # shared = GraphConvolutionLayer(6, adj, activation='linear')(shared)
    # shared = GraphConvolutionLayer(6, adj, activation='linear')(shared)
    # shared = GraphConvolutionLayer(6, adj, activation='linear')(shared)

    shared = tf.keras.layers.Flatten()(shared)

    # shared = tf.keras.layers.Dense(256, activation='relu')(shared)
    shared = tf.keras.layers.Dense(76, activation='linear')(shared)

    output = tf.keras.layers.Dense(1)(shared)
    model = tf.keras.Model(inputs=state_in, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.5,
                decay_steps=50,
                decay_rate=0.95,
            ),
            # learning_rate=rl_config['learning_rate']
        ),
        loss=tf.keras.losses.MeanSquaredError()
    )

    model.summary()

    for _ in (pbar := tqdm(range(20480), disable=False)):

        states = []
        rewards = []
        reward_from_state = []

        done = False
        obs = env.reset()
        states.append(obs)
        while not done:
            action = np.zeros(env.action_space.sample().shape)
            obs, reward, done, info = env.step(action)
            logging.getLogger('main').debug(f'Reward: {reward}')
            rewards.append([reward])
            states.append(obs)
            reward_from_state.append([get_reward_from_state(env, obs)])

        states = states[:-1]
        qs = [0 for _ in range(len(rewards))]
        qs[-1] = rewards[-1]
        for i in reversed(range(len(rewards) - 1)):
            qs[i] = [rewards[i][0] + rl_config['gamma'] * qs[i + 1][0]]

        states = np.array(states)
        qs = np.array(qs)
        # qs = np.expand_dims(qs, axis=1)
        actions = np.zeros((len(states), env.action_space.sample().shape[0], 1))

        pre_loss = model.criterion(qs, model(states))
        # pre_r2 = r2_score(qs, model(states), multioutput='variance_weighted')
        pre_r2 = r2(qs, model(states))

        previous_variables = model.trainable_variables[0].numpy()

        # for _ in range(rl_config['num_training_per_epoch']):
        #     with tf.GradientTape() as tape:
        #         q = model(states)
        #         loss = model.loss(qs, q)
        #     gradients = tape.gradient(loss, model.trainable_variables)
        #     model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # gradient_magnitude = tf.reduce_mean([tf.norm(g) for g in gradients])
        # print(f'Gradient magnitude: {gradient_magnitude:.2f}')
        # print(f'Loss: {loss:.2f}')

        model.fit(
            states,
            qs,
            epochs=rl_config['num_training_per_epoch'],
            verbose=0
        )

        # print('qs         =', end='')
        # print([f'{i:04.4f}' for i in list(qs.flatten())])
        # # print('model      =', end='')
        # # print([f'{i:04.4f}' for i in list(model.predict([states, actions], verbose=0).flatten())])
        # print('summing    =', end='')
        # print([f'{i:04.4f}' for i in list([i[0] for i in reward_from_state])])

        pbar.set_description(
            f'lr: {model.optimizer._decayed_lr(tf.float32):.3e} |'
            f' pre_loss: {pre_loss:.4f} |'
            f' loss: {model.evaluate(states, qs, verbose=0):.4f} |'
            f' loss_from_state = {model.criterion(qs, reward_from_state):.4f} |'
            f' pre_r2 = {pre_r2:.6f} | '
            # f' gradient_magnitude = {gradient_magnitude:.2f} |'
            # f' change(mean, sigma2) = ({(model.trainable_variables[0].numpy() - previous_variables).mean():.8f}, {(model.trainable_variables[0].numpy() - previous_variables).var():.8f}) |'
        )
