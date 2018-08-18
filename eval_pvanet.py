""" PVANet eval module
"""

import time
import six
import cifar_input
import numpy as np
import pvanet
import tensorflow as tf
import util
import config
slim = tf.contrib.slim


# ================================================== #
# Data Flags
# ================================================== #
tf.app.flags.DEFINE_string('dataset', 'cifar10', 'either cifar10 or cifar100.')
tf.app.flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.app.flags.DEFINE_integer('dataset_size', 10000, 'the size of evaluation data')

# ================================================== #
# Evaluation Flags
# ================================================== #
tf.app.flags.DEFINE_string('eval_data_path', '',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_dir', '', 'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('checkpoint_path', '', 'Directory if there are checkpoints to restore')
tf.app.flags.DEFINE_string('log_root', '', 'log root to restore checkpoints')
tf.app.flags.DEFINE_integer('num_gpus', 0,
                            'Number of gpus used for training. (0 or 1)')
tf.app.flags.DEFINE_integer('batch_size', 100, 'Batch size for test data')
tf.app.flags.DEFINE_integer('eval_batch_count', 50, 'the number of batches to evaluate')
tf.app.flags.DEFINE_bool('eval_once', False, 'if evaluate only once')
tf.app.flags.DEFINE_bool("ignore_missing_vars", False, '')
tf.app.flags.DEFINE_string('checkpoint_exclude_scopes', None, 'checkpoint_exclude_scopes')

tf.app.flags.DEFINE_float('weight_decay', 0.0005, 'The weight decay on the model weights.')

FLAGS = tf.app.flags.FLAGS

def config_initialization():
    # image shape and feature layers shape inference
    tf.logging.set_verbosity(tf.logging.DEBUG)
    util.proc.set_proc_name('test_pvanet')


def eval(images, labels):
    # Switching between network structures
    with slim.arg_scope(pvanet.pvanet_scope(is_training=False, weight_decay=FLAGS.weight_decay)):
        net, endpoints = pvanet.pvanet(images, num_classes=10,
                                       include_last_bn_relu=True,
                                       fatness=config.fatness,
                                       use_concat=config.concat,
                                       proj_type=config.proj_type)

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tf.train.start_queue_runners(sess)

    best_precision = 0.0
    while True:
        ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
        try:
            tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        except:
            import pdb
            pdb.set_trace()
        saver.restore(sess, ckpt_state.model_checkpoint_path)

        total_prediction, correct_prediction = 0, 0
        for _ in six.moves.range(FLAGS.eval_batch_count):
            (predictions, truth) = sess.run(
                [tf.nn.softmax(net), labels])

            truth = np.argmax(truth, axis=1)
            predictions = np.argmax(predictions, axis=1)
            correct_prediction += np.sum(truth == predictions)
            total_prediction += predictions.shape[0]

        precision = 1.0 * correct_prediction / total_prediction
        best_precision = max(precision, best_precision)

        precision_summ = tf.Summary()
        precision_summ.value.add(
            tag='Precision', simple_value=precision)
        summary_writer.add_summary(precision_summ)
        best_precision_summ = tf.Summary()
        best_precision_summ.value.add(
            tag='Best Precision', simple_value=best_precision)
        summary_writer.add_summary(best_precision_summ)
        tf.logging.info('precision: %.3f, best precision: %.3f' %
                        (precision, best_precision))
        summary_writer.flush()

        if FLAGS.eval_once:
            break

        time.sleep(60)


def main(_):
    config_initialization()
    images, labels = cifar_input.build_input(FLAGS.dataset, FLAGS.eval_data_path, FLAGS.batch_size, mode='eval')
    eval(images, labels)


if __name__ == '__main__':
    tf.app.run()
