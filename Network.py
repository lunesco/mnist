from abc import ABC
from functools import wraps

import tensorflow as tf


class layers:
    def __init__(self, seed=None, global_seed=None, initializer=tf.glorot_uniform_initializer, *args, **kwargs):
        self.initializer = initializer(seed=seed, *args, **kwargs)
        tf.compat.v1.random.set_random_seed(global_seed)

    @wraps(tf.layers.dense)
    def dense(self, *args, **kwargs):
        return tf.layers.dense(kernel_initializer=self.initializer, *args, **kwargs)


class AbstractNetwork(ABC):

    def __init__(self, placeholders=None, activation_function='relu', architecture=(784, 784, 784, 784),
                 initializer_seed=None, global_seed=None):
        self._placeholders = placeholders
        self._architecture = architecture
        self._initializer_seed = initializer_seed
        self.layers = layers(initializer_seed, global_seed)

        if activation_function == 'relu':
            self._activation_function = tf.nn.relu
        elif activation_function == 'leaky_relu':
            self._activation_function = tf.nn.leaky_relu
        elif activation_function == 'tanh':
            self._activation_function = tf.nn.tanh
        elif activation_function == 'sigmoid':
            self._activation_function = tf.nn.sigmoid
        else:
            raise Exception("Bad activation function")

        self._output = None

    @property
    def output(self):
        return self._output


class ClassifierNetwork(AbstractNetwork):

    def __init__(self, placeholders=None, num_classes=10, drop_prob=0.4, activation_function='relu',
                 architecture=(784, 784, 784, 784), initializer_seed=None, dropout_seed=None, global_seed=None):
        super().__init__(placeholders, activation_function, architecture, initializer_seed, global_seed)
        self._num_classes = num_classes
        self._drop_prob = drop_prob
        self._dropout_seed = dropout_seed
        self._output = self.neural_net(*self._architecture)

    def neural_net(self, *args):

        net_input = self._placeholders.x
        neurons_in_certain_layer = []
        for number_of_neurons in args:
            neurons_in_certain_layer.append(number_of_neurons)

        with tf.variable_scope('Neural_Network'):
            for num_units in neurons_in_certain_layer:
                net_input = self.layers.dense(inputs=net_input, units=num_units, activation=self._activation_function)

        with tf.variable_scope('Output_layer'):
            # Add dropout operation; 0.6 probability that element will be kept
            dropout = tf.layers.dropout(inputs=net_input, rate=self._drop_prob, seed=self._dropout_seed,
                                        training=self._placeholders.is_training)
            # Logits layer - Output Tensor Shape: [batch_size, 10]
            output_layer = self.layers.dense(inputs=dropout, units=self._num_classes)

        return output_layer


class Autoencoder(AbstractNetwork):

    def __init__(self, placeholders=None, activation_function='relu', architecture=(8,), initializer_seed=None,
                 latent_activation='tanh', output_layer_activation='sigmoid', global_seed=None):
        super().__init__(placeholders, activation_function, architecture, initializer_seed, global_seed)
        self._latent_activation = latent_activation
        self._output_layer_activation = output_layer_activation
        self._latent, self._output = self.neural_net(*self._architecture)

    @property
    def latent(self):
        return self._latent

    @property
    def latent_activation(self):
        return self._latent_activation

    def neural_net(self, *args):

        layer_input = self._placeholders.x
        placeholder_size = self._placeholders.x.get_shape().as_list()
        input_size = placeholder_size[-1]

        num_hidden = []
        for neurons in args:
            num_hidden.append(neurons)

        with tf.variable_scope('Encoder'):
            for num_units in num_hidden[:-1]:
                layer_input = self.layers.dense(inputs=layer_input, units=num_units,
                                                activation=self._activation_function, name='encoder_layer')

        with tf.variable_scope("Latent_layer"):
            layer_input = self.layers.dense(inputs=layer_input, units=num_hidden[-1], activation=self.latent_activation,
                                            name='latent_layer')
            latent = tf.identity(layer_input)

        with tf.variable_scope('Decoder'):
            num_hidden_decoder = list(reversed(num_hidden))[1:] + [input_size]
            for num_units_dec in num_hidden_decoder[:-1]:
                layer_input = self.layers.dense(inputs=layer_input, units=num_units_dec,
                                                activation=self._activation_function, name='decoder_layer')

        with tf.variable_scope("Reconstruction"):
            layer_input = self.layers.dense(inputs=layer_input, units=num_hidden_decoder[-1],
                                            activation=self._output_layer_activation, name='reconstruction_layer')
            output_layer = tf.identity(layer_input)

        return latent, output_layer
