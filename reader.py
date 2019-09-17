import math
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import dtypes

DOWNLOAD_DIRNAME = "resources/"


class DbReader:
    def __init__(self, data_type=dtypes.float32, boolean_threshold=0.5, one_hot=True, seed=None):
        """
        :param data_type: one of tensorflow.python.framework.dtypes.[bool | uint8 | float32]
        :param boolean_threshold: sets the threshold for binarization [0, 1] (applies only if :param data_type is set to
        'bool', otherwise ignored
        :param one_hot: sets the coding for labels (one hot or integers) [bool]
        """
        print('Reading dataset...')
        dtype = data_type
        if data_type == dtypes.bool:
            dtype = dtypes.float32
        mnist = input_data.read_data_sets(DOWNLOAD_DIRNAME, dtype=dtype, one_hot=one_hot)
        self.random_state = np.random.RandomState(seed)
        self._train_images = mnist.train.images
        self._train_labels = mnist.train.labels
        self._valid_images = mnist.validation.images
        self._valid_labels = mnist.validation.labels
        self._test_images = mnist.test.images
        self._test_labels = mnist.test.labels
        if data_type == dtypes.bool:
            self._convert_to_bool(boolean_threshold)
        print('Done')

    def _convert_to_bool(self, boolean_threshold):
        self._train_images = (self._train_images >= boolean_threshold).astype(np.bool_)
        self._valid_images = (self._valid_images >= boolean_threshold).astype(np.bool_)
        self._test_images = (self._test_images >= boolean_threshold).astype(np.bool_)

    def get_train_set(self, number_of_samples=None):
        """
        method returns features (images) and labels for training data
        :return: (np.ndarray, np.ndarray): (images, labels)
        """
        return self.get_samples(self._train_images, self._train_labels, number_of_samples)

    def get_validation_set(self, number_of_samples=None):
        """
        method returns features (images) and labels for validation data
        :return: (np.ndarray, np.ndarray): (images, labels)
        """
        return self.get_samples(self._valid_images, self._valid_labels, number_of_samples)

    def get_test_set(self, number_of_samples=None):
        """
        method returns features (images) and labels for test data
        :return: (np.ndarray, np.ndarray): (images, labels)
        """
        return self.get_samples(self._test_images, self._test_labels, number_of_samples)

    def get_samples(self, x, y, number_of_samples):
        if number_of_samples is not None:
            total_size = x.shape[0]
            random_indices = self.random_state.choice(total_size, number_of_samples, replace=False)
            return x[random_indices], y[random_indices]
        else:
            return x, y

    @staticmethod
    def get_batches(images, labels, batch_size):
        """
        method splits the data :param images and :param labels into batches of size :param batch_size. If
        data_len%batch_size != 0 then the last batch is smaller than the rest
        :param images: np.ndarray with features (images)
        :param labels: np.ndarray with labels
        :param batch_size: desired batch size
        :return: generator returning a tuple (np.ndarray, np.ndarray) with batch (images, labels)
        """
        data_len = images.shape[0]
        num_batches = math.ceil(data_len / batch_size)
        for i in range(num_batches):
            yield (images[i * batch_size:(i + 1) * batch_size], labels[i * batch_size:(i + 1) * batch_size])
