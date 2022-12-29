from multiprocessing.pool import Pool

import numpy as np


def get_q_model(loss_function, optimizer):
    import tensorflow as tf

    action_shape = (76, )
    state_shape = (76, 2)

    state_in = tf.keras.layers.Input(shape=state_shape)
    action_in = tf.keras.layers.Input(shape=action_shape)
    action_reshaped = tf.keras.layers.Reshape((action_shape[0], 1))(action_in)

    mix = tf.keras.layers.Concatenate(axis=2)([state_in, action_reshaped])


    shared = tf.keras.layers.Flatten()(mix)
    shared = tf.keras.layers.Dense(8, activation='relu')(shared)

    output = tf.keras.layers.Dense(1)(shared)
    model = tf.keras.Model(inputs=[state_in, action_in], outputs=output)

    model.compile(
        optimizer=optimizer,
        loss=loss_function,
    )

    model.summary()

    return model


def get_optimal_action_and_value(states, action_dim, model, action_gradient_step_count, action_optimizer_lr, norm,
                                 epsilon):
    import tensorflow as tf
    actions = tf.Variable(tf.random.normal((states.shape[0], action_dim)), name='action')
    print(actions.shape)
    print(states.shape)
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

        histogram[i] = tf.reduce_mean((model([states, actions]).numpy() - before.numpy()) / before.numpy()) * 100

    q_values = model([states, actions])

    return actions, q_values, tf.convert_to_tensor(histogram)


def run_multi_process(states, model):
    for _ in range(1000):
        get_optimal_action_and_value(states, 76, model, 20, 0.1, 2, 1)
    return True


if __name__ == '__main__':
    with Pool(4) as tp:
        tp.starmap(run_multi_process, [(np.random.rand(512, 76, 2), get_q_model('mse', 'adam')) for _ in range(12)])
    # run_multi_process(np.random.rand(512, 76, 2), get_q_model('mse', 'adam'))