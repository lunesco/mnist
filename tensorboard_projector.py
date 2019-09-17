import os
import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector

EMBEDDING_FILENAME = 'embeddings.ckpt'
EMBEDDING_LIMIT = 1e5
DEFAULT_LOGS_DIR = './logs/projector/'

# EMBEDDING_FILENAME = 'embed'
# EMBEDDING_LIMIT = 50000


class EvalMode(object):
    ORIGINAL = 'original'
    CLASS = 'class'
    PROBAB = 'probab'
    LATENT = 'latent'


class Projector:
    @staticmethod
    def prepare_tensorboard_projector(classes,
                                      latent,
                                      source_file_list,
                                      input_signal,
                                      log_dir=DEFAULT_LOGS_DIR):
        # type: (list, list, list, list, str) -> None
        """Parses signal array and auxiliary information signals into TensorBoard Projector files.

        TensorBoard Projector can be used to compute PCA or t-SNE representation of input array.

        :param classes: list; ground truth class label for each example
        :param latent: list; array to be visualized
        :param source_file_list: list; source filename for each example
        :param input_signal: list; complementary array to latent, containing network input for each example
        :param log_dir: path str; directory where TensorBoard files will be saved
        """

        # reset graph
        try:
            tf.reset_default_graph()
        except AssertionError as err:
            # do not clear default graph if we're nested in an explicit tf.Graph() instance
            print(err)

        # prepare datasets
        classes = np.asarray(classes)
        latent = np.asarray(latent)
        source_file_list = np.asarray(source_file_list)
        input_signal = np.asarray(input_signal)

        # shuffle dataset (Tensorboard Embedding Projector has a limit of 100k samples)
        random_state = np.random.RandomState(12455)
        random_indexes = list(range(len(latent)))
        random_state.shuffle(random_indexes)

        latent = latent[random_indexes]
        classes = classes[random_indexes]
        source_file_list = source_file_list[random_indexes]
        if input_signal[0] is not None:
            input_signal = input_signal[random_indexes]

        # flatten image to 1D
        if len(latent.shape) > 2:
            latent = latent.reshape(latent.shape[0], -1)

        # add projector
        summary_writer = tf.summary.FileWriter(log_dir)
        config = projector.ProjectorConfig()

        # split samples into multiple embeddings
        embedding_limit = int(EMBEDDING_LIMIT)
        num_embeddings = len(latent) // embedding_limit + 1

        # create embeddings separately for latent layer and inputs (if present)
        if input_signal[0] is not None:
            projections = [latent, input_signal]
            projection_names = [EvalMode.LATENT, EvalMode.ORIGINAL]
        else:
            projections = [latent]
            projection_names = [EvalMode.LATENT]

        for proj_idx in range(len(projections)):
            signal = projections[proj_idx]
            embedding_name_base = projection_names[proj_idx]

            for idx in range(num_embeddings):
                indices = slice(idx * embedding_limit, min(len(signal), (idx + 1) * embedding_limit), 1)

                metadata_file = embedding_name_base + '_{}'.format(idx) + '.tsv'
                embedding_name = embedding_name_base + '_{}'.format(idx)

                # assign embeddings
                embedding_var = tf.Variable(signal[indices, :], name=embedding_name)

                # write label info
                metadata_path = os.path.join(log_dir, metadata_file)
                with open(metadata_path, 'w') as f:
                    f.write("Class\tDatabase\tFilename\n")
                    for fragment_idx in range(len(classes[indices])):
                        label = int(classes[indices][fragment_idx])
                        filename = source_file_list[indices][fragment_idx]
                        filename = os.path.splitext(os.path.basename(filename))[0]
                        database = filename.split('_')[0]
                        f.write("{}\t{}\t{}\n".format(label, database, filename))

                embedding = config.embeddings.add()
                embedding.tensor_name = embedding_var.name
                embedding.metadata_path = metadata_file

        # save projection
        projector.visualize_embeddings(summary_writer, config)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(log_dir, EMBEDDING_FILENAME), 1)
