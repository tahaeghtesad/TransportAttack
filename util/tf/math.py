import tensorflow as tf


# @tf.function
def r2(x, y):
    # assert x.shape == y.shape, f'x and y must have the same shape, but got {x.shape} and {y.shape}'
    return 1 - tf.reduce_sum((y - tf.reduce_mean(y)) ** 2)/tf.reduce_sum((y - x) ** 2)


# @tf.function
def cdist(x, y):
    per_x_dist = lambda i: tf.norm(x[:, i:(i + 1), :] - y, axis=2)
    dist = tf.map_fn(fn=per_x_dist, elems=tf.range(tf.shape(x)[1], dtype=tf.int64), fn_output_signature=x.dtype)
    return tf.transpose(dist, perm=[1, 0, 2])
