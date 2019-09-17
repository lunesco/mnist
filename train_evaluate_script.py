import argparse

import tensorflow as tf
from tensorflow.python.framework import dtypes

from Fetch import Fetch
from Network import ClassifierNetwork
from Placeholders import Placeholders
from reader import DbReader
from tensorboard_plotting import Plotter

N_OF_CLASSES = 10
LEARNING_RATE = 0.001
RANDOM_SEED_NETWORK_DROPOUT = 12345
RANDOM_SEED_NETWORK = 12345
RANDOM_SEED_GLOBAL = 12345


# tensorboard usage: in project main dir in cmd: tensorboard --logdir=./logs/
# in browser: localhost:6006
def main(data_type, batch_size, n_epochs, n_of_classes, learning_rate, drop_prob, activation_function, num_neurons,
         cost_function, optimizer, dropout_seed, initializer_seed, global_seed):
    max_acc_valid = 0
    reader = DbReader(data_type=data_type)
    X_train, y_train = reader.get_train_set()
    # X_test, y_test = reader.get_test_set()
    n_of_features = X_train.shape[1]
    placeholders = Placeholders(n_of_features, n_of_classes)
    network = ClassifierNetwork(placeholders, num_classes=n_of_classes, drop_prob=drop_prob,
                                initializer_seed=initializer_seed,
                                activation_function=activation_function, architecture=num_neurons,
                                dropout_seed=dropout_seed, global_seed=global_seed)
    fetches = Fetch(network.output, placeholders, loss=cost_function, optimizer=optimizer)

    X_valid, y_valid = reader.get_validation_set()
    reset_metrics_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

    with tf.Session() as sess:
        writer = Plotter(sess.graph)

        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        histogram = sess.run(fetches.update_histogram)
        writer.write_histogram(histogram)

        for epoch in range(n_epochs):
            for X_batch, y_batch in reader.get_batches(X_train, y_train, batch_size):
                sess.run(fetches.update_metrics_and_weights(),
                         feed_dict=placeholders.feed_dict(X_batch, y_batch, learning_rate=learning_rate,
                                                          is_training=True))

            error_train, acc_train, histogram = sess.run(fetches.get_metrics() + [fetches.update_histogram])
            writer.write_train_summary(error_train, epoch, acc_train)
            writer.write_histogram(histogram)
            sess.run(reset_metrics_op)

            for X_batch, y_batch in reader.get_batches(X_valid, y_valid, batch_size):
                sess.run(fetches.update_metrics(), feed_dict=placeholders.feed_dict(X_batch, y_batch,
                                                                                    is_training=False))
            error_valid, acc_valid = sess.run(fetches.get_metrics())
            writer.write_valid_summary(error_valid, epoch, acc_valid)

            print(
                'epoch: {}, error for training set: {}, error for validation set: {},'
                ' train_acc for training set: {},'
                ' train_acc for validation set: {}'.format(
                    epoch, error_train,
                    error_valid, acc_train, acc_valid))

            if max_acc_valid < acc_valid:
                max_acc_valid = acc_valid

            sess.run(reset_metrics_op)

        print("Maximal acccuracy on valid dataset: {0:.5f}".format(max_acc_valid))
        with open('results_2.txt', 'a+') as f:
            f.write(
                "\n \n Error train: {}, Error valid: {}, Train acc: {}, Valid acc: {}".format(error_train, error_valid,
                                                                                              acc_train, acc_valid))
            f.write("\n Max accuracy on valid: {}".format(max_acc_valid))
            f.write(
                "\n Batch size:{}, Epochs: {}, Learning rate: {}, Drop prob: {}, Activation function: {}, Neurons: {}, Cost function: {}, Optimizer: {}".format(
                    batch_size, n_epochs, learning_rate, drop_prob, activation_function, num_neurons, cost_function,
                    optimizer))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural network training and evaluating for MNIST dataset')

    parser.add_argument('--data_type', default=dtypes.float32, choices=[dtypes.float32, dtypes.uint8, dtypes.bool])
    parser.add_argument('--batch_size', default=1000, type=int, help='batch size for feeding data into neural '
                                                                     'network')
    parser.add_argument('--n_epochs', default=10, type=int, help='number of epochs')
    parser.add_argument('--initializer_seed', default=RANDOM_SEED_NETWORK, type=int,
                        help='seed for weights initialization')
    parser.add_argument('--dropout_seed', default=RANDOM_SEED_NETWORK, type=int, help='seed for dropout')
    parser.add_argument('--global_seed', default=RANDOM_SEED_GLOBAL, type=int, help='global seed for graph')

    parser.add_argument('--n_of_classes', default=N_OF_CLASSES, type=int, help='number of classes')
    parser.add_argument('--learning_rate', default=LEARNING_RATE, type=float)

    parser.add_argument('--drop_prob', default=0.4, type=float, help='Probability of dropping a neurons')
    parser.add_argument('--activation_function', default='relu', type=str, help='Neuron activation function')
    # Proper way of adding --num_neurons argument: --num_neurons 784 784 784 784
    parser.add_argument('--num_neurons', default=(784, 784,), nargs='+', type=int, help='Neurons in certain layer')

    parser.add_argument('--cost_function', default='softmax', type=str, help='Type of cost function')
    parser.add_argument('--optimizer', default='sgd', type=str, help='Type of optimizer')

    args = vars(parser.parse_args())
    main(**args)
