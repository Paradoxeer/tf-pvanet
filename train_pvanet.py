"""PVANet Train module.
"""
import time
import cifar_input
import pvanet
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import tf_logging as logging
import util
import config
slim = tf.contrib.slim


# ================================================== #
# Data Flags
# ================================================== #
tf.app.flags.DEFINE_string('dataset', 'cifar10', 'either cifar10 or cifar100.')
tf.app.flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.app.flags.DEFINE_integer('dataset_size', 50000, 'the size of training data')

# ================================================== #
# Training Flags
# ================================================== #
tf.app.flags.DEFINE_string('train_data_path', '',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('train_dir', '', 'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('checkpoint_path', '', 'Directory if there are checkpoints to restore')
tf.app.flags.DEFINE_integer('num_gpus', 0,
                            'Number of gpus used for training. (0 or 1)')
tf.app.flags.DEFINE_integer('max_train_steps', 100000,
                            'Number of maximum training steps')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size for training data')
tf.app.flags.DEFINE_bool("ignore_missing_vars", False, '')
tf.app.flags.DEFINE_string('checkpoint_exclude_scopes', None, 'checkpoint_exclude_scopes')
tf.app.flags.DEFINE_integer('log_every_n_steps', 1, 'log frequency')

# ================================================== #
# Optimizer Flags
# ================================================== #
tf.app.flags.DEFINE_float('learning_rate', 0.1, 'initial learning rate')
tf.app.flags.DEFINE_string('learning_rate_type', 'exponential', 'the type of learning rate')
tf.app.flags.DEFINE_float('exponential_decay_rate', 0.1, 'the decay rate of exponential learning rate')
tf.app.flags.DEFINE_integer('decay_step', 40000, 'the decay step for learning rate')
tf.app.flags.DEFINE_float('min_learning_rate', 0.0001, 'the minimum learning rate allowed')

tf.app.flags.DEFINE_string('optimizer', 'mom', 'type of gradient descent optimizer to use, mom/sgd/adam')
tf.app.flags.DEFINE_float('momentum', 0.9, 'The momentum for the MomentumOptimizer')

tf.app.flags.DEFINE_float('weight_decay', 0.0002, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_bool('using_moving_average', False, 'Whether to use ExponentionalMovingAverage')
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, 'The decay rate of ExponentionalMovingAverage')

FLAGS = tf.app.flags.FLAGS


def config_initialization():
    # image shape and feature layers shape inference
    image_shape = (FLAGS.image_size, FLAGS.image_size)

    if not FLAGS.train_data_path:
        raise ValueError('You must supply the dataset directory with --train_data_path')

    tf.logging.set_verbosity(tf.logging.DEBUG)
    util.init_logger(
        log_file='log_cifar10_pvanet_%d_%d.log' % image_shape,
        log_path=FLAGS.train_dir, stdout=False, mode='a')

    batch_size = FLAGS.batch_size
    tf.summary.scalar('batch_size', batch_size)
    util.proc.set_proc_name('pvanet_cifar10')


def create_train_op(images, labels):
    with tf.device('/cpu:0'):
        global_step = slim.create_global_step()

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    # Switching between network structures:
    print('using regular pvanet')
    with slim.arg_scope(pvanet.pvanet_scope(is_training=True, weight_decay=FLAGS.weight_decay)):
        logits, endpoints = pvanet.pvanet(images, num_classes=10,
                                          include_last_bn_relu=True,
                                          fatness=config.fatness,
                                          use_concat=config.concat,
                                          proj_type=config.proj_type)

    losses = slim.losses.softmax_cross_entropy(logits, labels, scope='softmax_loss')
    losses_reduced = tf.reduce_mean(losses)

    correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Add summaries for variables.
    for variable in slim.get_model_variables():
        summaries.add(tf.summary.histogram(variable.op.name, variable))

    with tf.device('/cpu:0'):
        if FLAGS.learning_rate_type == 'exponential':
            decay_steps = FLAGS.decay_step
            learning_rate = tf.maximum(tf.train.exponential_decay(FLAGS.learning_rate,
                                                   global_step=global_step,
                                                   decay_steps=decay_steps,
                                                   decay_rate=FLAGS.exponential_decay_rate,
                                                   staircase=True),
                                       tf.constant(FLAGS.min_learning_rate))
        elif FLAGS.learning_rate_type == 'constant':
            learning_rate = tf.constant(FLAGS.learning_rate, name='constant_learning_rate')

        if FLAGS.optimizer == 'mom':
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=FLAGS.momentum, name='mom')
        elif FLAGS.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate, name='sgd')
        elif FLAGS.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate,
                                               beta1=config.beta1, beta2=config.beta2, epsilon=config.epsilon,
                                               name='adam')
        summaries.add(tf.summary.scalar('learning_rate', learning_rate))

    with tf.device('/gpu:0'):
        # Get losses and gradients
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        sum_loss = tf.add_n([losses_reduced] + regularization_losses)

    grad = optimizer.compute_gradients(sum_loss)
    summaries.add(tf.summary.scalar('total_loss', sum_loss))

    grad_updates = optimizer.apply_gradients(grad, global_step=global_step)
    train_ops = [grad_updates]

    bn_update_op = util.tf.get_update_op()
    if bn_update_op is not None:
        train_ops.append(bn_update_op)

    # Configure the moving averages #
    if FLAGS.using_moving_average:
        tf.logging.info('using moving average for train with decay %f'%FLAGS.moving_average_decay)
        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
        ema_op = variable_averages.apply(tf.trainable_variables())
        with tf.control_dependencies([grad_updates]):
            train_ops.append(tf.group(ema_op))
    else:
        moving_average_variables, variable_averages = None, None

    train_ops = control_flow_ops.with_dependencies(train_ops, sum_loss, name='train_op')

    def train_step_fn(sess, train_ops, global_step, train_step_kwargs):
        start_time = time.time()
        total_loss, np_global_step = sess.run([train_ops, global_step])
        precision = sess.run(accuracy)
        time_elapsed = time.time() - start_time

        if 'should_log' in train_step_kwargs:
            if sess.run(train_step_kwargs['should_log']):
                logging.info('global step %d: loss = %.4f ; precision = %.4f (%.3f sec/step)',
                             np_global_step, total_loss, precision, time_elapsed)

        if 'should_stop' in train_step_kwargs:
            should_stop = sess.run(train_step_kwargs['should_stop'])
        else:
            should_stop = False
        return total_loss, should_stop

    return train_ops, train_step_fn


def train(train_op, train_step_fn):
    summary_op = tf.summary.merge_all()
    sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)

    init_fn = util.tf.get_init_fn(checkpoint_path=FLAGS.checkpoint_path, train_dir=FLAGS.train_dir,
                                  ignore_missing_vars=FLAGS.ignore_missing_vars,
                                  checkpoint_exclude_scopes=FLAGS.checkpoint_exclude_scopes)
    saver = tf.train.Saver(max_to_keep=500, write_version=2)
    slim.learning.train(
        train_op,
        train_step_fn=train_step_fn,
        logdir=FLAGS.train_dir,
        init_fn=init_fn,
        summary_op=summary_op,
        number_of_steps=FLAGS.max_train_steps,
        log_every_n_steps=FLAGS.log_every_n_steps,
        save_summaries_secs=300,
        saver=saver,
        save_interval_secs=3600,
        session_config=sess_config
    )


def main(_):
    config_initialization()
    images, labels = cifar_input.build_input(FLAGS.dataset, FLAGS.train_data_path, FLAGS.batch_size, mode='train')
    train_op, train_step_fn = create_train_op(images, labels)
    train(train_op, train_step_fn)

if __name__=='__main__':
    tf.app.run()












