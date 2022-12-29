import tensorflow as tf


class LNormOptimizerLayer(tf.keras.layers.Layer):
    def __init__(self, ord, length, **kwargs):
        super(LNormOptimizerLayer, self).__init__(**kwargs)
        self.ord = ord
        self.length = length

    def build(self, input_shape):
        super(LNormOptimizerLayer, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        norm = tf.norm(inputs, ord=self.ord, axis=1, keepdims=True)
        return self.length * tf.math.divide_no_nan(inputs, norm)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super(LNormOptimizerLayer, self).get_config()