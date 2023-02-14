import tensorflow as tf

@tf.function
def r2(x, y):
    assert x.shape == y.shape, f'x and y must have the same shape, but got {x.shape} and {y.shape}'
    return 1 - tf.reduce_sum((y - tf.reduce_mean(y)) ** 2)/tf.reduce_sum((y - x) ** 2)