import tensorflow as tf


class GraphConvolutionLayer(tf.keras.layers.Layer):
    def __init__(self, units, adj, activation=None, **kwargs):
        super(GraphConvolutionLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.adj = adj

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=(input_shape[-1], self.units), initializer='uniform', trainable=True)
        self.bias = self.add_weight(name='bias', shape=(self.units,), initializer='uniform', trainable=True)
        self.beta = self.add_weight(name='beta', shape=(self.adj.shape[0],), initializer='uniform', trainable=True)
        super(GraphConvolutionLayer, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):

        output = tf.linalg.diag(tf.math.reciprocal(tf.reduce_sum(self.adj, axis=1) + self.beta)) @ (self.adj + tf.linalg.diag(self.beta)) @ inputs @ self.kernel + self.bias

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def get_config(self):
        config = super(GraphConvolutionLayer, self).get_config()
        config.update({
            'units': self.units,
            'activation': self.activation,
            'adj': self.adj
        })
        return config