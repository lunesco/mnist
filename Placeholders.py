import tensorflow as tf


class Placeholders(object):
    def __init__(self, number_of_features, number_of_classes):
        with tf.name_scope('placeholders'):
            self._learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
            self._is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
            self._x = tf.placeholder(tf.float32, shape=(None, number_of_features), name='x')
            self._y = tf.placeholder(tf.float32, shape=(None, number_of_classes), name='y')

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def is_training(self):
        return self._is_training

    @property
    def learning_rate(self):
        return self._learning_rate

    def feed_batch(self, x_batch, y_batch):
        return {self.x: x_batch, self.y: y_batch}

    def feed_is_training(self, is_training):
        return {self.is_training: is_training}

    def feed_learning_rate(self, learning_rate):
        return {self.learning_rate: learning_rate}

    def feed_dict(self, x_batch=None, y_batch=None, is_training=None, learning_rate=None):
        all_params = {**self.feed_batch(x_batch, y_batch),
                      **self.feed_is_training(is_training),
                      **self.feed_learning_rate(learning_rate)}
        dict_without_none = {k: v for k, v in all_params.items() if v is not None}
        return dict_without_none
