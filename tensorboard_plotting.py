import datetime

import tensorflow as tf


class Plotter:
    def __init__(self, graph):
        self._train_filename = "./logs/train/" + str(datetime.datetime.today()).replace(' ', '-').replace('.',
                                                                                                          '-').replace(
            ':', '-')
        self._valid_filename = "./logs/valid/" + str(datetime.datetime.today()).replace(' ', '-').replace('.',
                                                                                                          '-').replace(
            ':', '-')
        self._train_writer = tf.summary.FileWriter(self._train_filename, graph)
        self._valid_writer = tf.summary.FileWriter(self._valid_filename, graph)

    @staticmethod
    def _write_summary(writer, tag, value, epoch):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        writer.add_summary(summary, epoch)

    def write_train_summary(self, error, epoch, acc=None):
        self._write_summary(self._train_writer, 'error', error, epoch)
        if acc is not None:
            self._write_summary(self._train_writer, 'accuracy', acc, epoch)

    def write_valid_summary(self, error, epoch, acc=None):
        self._write_summary(self._valid_writer, 'error', error, epoch)
        if acc is not None:
            self._write_summary(self._valid_writer, 'accuracy', acc, epoch)

    def write_histogram(self, histogram_summary):
        self._train_writer.add_summary(histogram_summary)
