import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorboard_plotting import Plotter
from Fetch import AutoencoderFetch
from Network import Autoencoder
from Placeholders import Placeholders
from reader import DbReader
from tensorboard_projector import Projector

# tensorboard projector usage: tensorboard --logdir ./logs/projector/

LEARNING_RATE = 0.001
WIDTH = 7
PRECISION = 4
RANDOM_SAMPLE_SIZE = 2000
RANDOM_SEED_NETWORK = 12345
RANDOM_SEED_SAMPLER = 12345
RANDOM_SEED_GLOBAL = 12345


def plot_two_dimension_latent_space(latent_space, labels, title, ax=None):
    data = pd.DataFrame({'z1': latent_space[:, 0], 'z2': latent_space[:, 1], 'label': labels})
    sns.scatterplot('z1', 'z2', hue='label', legend="full", palette=sns.color_palette("hls", 10), data=data, ax=ax)
    plt.title(title)


def plot_results(network_input, network_output, title):
    f, a = plt.subplots(2, 10, figsize=(12, 6))
    plt.suptitle(title)
    for i, x in enumerate(network_input):
        a[0][i].imshow(np.reshape(x, (28, 28)))
        a[1][i].imshow(np.reshape(network_output[i], (28, 28)))
    plt.show()


def autoencoder(data_type, batch_size, n_epochs, learning_rate, activation_function, num_neurons, cost_function,
                optimizer, sample_size, latent_layer_activation, initializer_seed, global_seed, reader_seed,
                output_layer_activation=None):
    reader = DbReader(data_type=data_type, one_hot=False, seed=reader_seed)
    X_train, y_train = reader.get_train_set()
    X_valid, y_valid = reader.get_validation_set()

    X_train_sampled, y_train_sampled = reader.get_train_set(sample_size)
    X_valid_sampled, y_valid_sampled = reader.get_validation_set(sample_size)

    n_of_features = X_train.shape[1]
    placeholders = Placeholders(n_of_features, n_of_features)
    network = Autoencoder(placeholders, activation_function=activation_function, architecture=num_neurons,
                          latent_activation=latent_layer_activation, output_layer_activation=output_layer_activation,
                          initializer_seed=initializer_seed, global_seed=global_seed)
    fetches = AutoencoderFetch(network.output, placeholders, loss=cost_function, optimizer=optimizer)

    reset_metrics_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        writer = Plotter(sess.graph)
        histogram = sess.run(fetches.update_histogram)
        writer.write_histogram(histogram)
        for epoch in range(n_epochs):
            for X_batch, _ in reader.get_batches(X_train, y_train, batch_size):
                sess.run(fetches.update_metrics_and_weights(),
                         feed_dict=placeholders.feed_dict(X_batch, learning_rate=learning_rate,
                                                          is_training=True))

            error_train, histogram = sess.run([fetches.get_metrics(), fetches.update_histogram])
            writer.write_train_summary(error_train, epoch)
            writer.write_histogram(histogram)
            sess.run(reset_metrics_op)

            for X_batch, _ in reader.get_batches(X_valid, y_valid, batch_size):
                sess.run(fetches.update_metrics(), feed_dict=placeholders.feed_dict(X_batch, is_training=False))

            error_valid = sess.run(fetches.get_metrics())
            writer.write_valid_summary(error_valid, epoch)

            print(f'epoch: {epoch}, error for training set: {error_train:{WIDTH}.{PRECISION}}, '
                  f'error for validation set: {error_valid:{WIDTH}.{PRECISION}}')
            sess.run(reset_metrics_op)

        latent_space_train, results_train = sess.run([network.latent, network.output],
                                                     feed_dict=placeholders.feed_dict(X_train_sampled))
        latent_space_valid, results_valid = sess.run([network.latent, network.output],
                                                     feed_dict=placeholders.feed_dict(X_valid_sampled))

        latent_full_training = np.ndarray((0, num_neurons[-1]))
        for X_batch, _ in reader.get_batches(X_train, y_train, batch_size):
            latent_training_batch = sess.run(network.latent, feed_dict=placeholders.feed_dict(X_batch))
            latent_full_training = np.concatenate((latent_full_training, latent_training_batch), axis=0)

        if num_neurons[-1] == 2:
            plt.figure()
            plt.subplot(1, 2, 1)
            plot_two_dimension_latent_space(latent_space_train, y_train_sampled, title='train')
            plt.subplot(1, 2, 2)
            plot_two_dimension_latent_space(latent_space_valid, y_valid_sampled, title='valid')
            plt.show()
        # Comparing original images with reconstructions
        plot_results(X_train_sampled[:10], results_train[:10], title='train')
        plot_results(X_valid_sampled[:10], results_valid[:10], title='valid')

    Projector.prepare_tensorboard_projector(y_train, latent_full_training, ['mnist' for _ in y_train], X_train)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural network training and evaluating for MNIST dataset')

    parser.add_argument('--data_type', default=dtypes.float32, choices=[dtypes.float32, dtypes.uint8, dtypes.bool])
    parser.add_argument('--batch_size', default=1000, type=int, help='batch size for feeding data into neural '
                                                                     'network')
    parser.add_argument('--n_epochs', default=50, type=int, help='number of epochs')

    parser.add_argument('--initializer_seed', default=RANDOM_SEED_NETWORK, type=int,
                        help='seed for weights initialization')
    parser.add_argument('--reader_seed', default=RANDOM_SEED_SAMPLER, type=int, help='seed for sampling from dataset')
    parser.add_argument('--global_seed', default=RANDOM_SEED_GLOBAL, type=int, help='global seed for graph')

    parser.add_argument('--learning_rate', default=LEARNING_RATE, type=float)

    parser.add_argument('--activation_function', default='relu', choices=['sigmoid', 'tanh', 'relu', 'elu', 'linear'],
                        type=str, help='Neuron activation function')
    # Proper way of adding --num_neurons argument: --num_neurons 784 784 784 784
    parser.add_argument('--num_neurons', default=(200, 2), nargs='+', type=int,
                        help='Neurons in certain layer')

    parser.add_argument('--cost_function', default='mse', choices=['softmax', 'hinge', 'mse', 'bce'],
                        type=str, help='Type of cost function')
    parser.add_argument('--optimizer', default='Adam', type=str, help='Type of optimizer')
    parser.add_argument('--sample_size', default=RANDOM_SAMPLE_SIZE, type=int, help='number of samples for plot')

    # used for better latent (2D plot) visualization: tanh
    parser.add_argument('--latent_layer_activation', default='relu',
                        choices=['sigmoid', 'tanh', 'relu', 'elu', 'linear'],
                        type=str, help='Latent layer activation')

    # used for better output_layer visualization: sigmoid
    parser.add_argument('--output_layer_activation', default=None,
                        choices=['sigmoid', 'tanh', 'relu', 'elu', 'linear'],
                        type=str, help='Output layer activation')

    args = vars(parser.parse_args())
    autoencoder(**args)
