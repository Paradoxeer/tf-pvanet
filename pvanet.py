"""
Modified from dengdan: https://github.com/dengdan/tf-pvanet/blob/master/pvanet.py
"""

from collections import namedtuple
import tensorflow as tf

slim = tf.contrib.slim

BLOCK_TYPE_MCRELU = 'BLOCK_TYPE_MCRELU'
BLOCK_TYPE_INCEP = 'BLOCK_TYPE_INCEP'

BlockConfig = namedtuple('BlockConfig',
                         'stride, num_outputs, preact_bn, block_type, proj_type')


def __conv(net, kernel_size, stride, num_outputs, scope='conv'):
    net = slim.conv2d(inputs=net,
                      num_outputs=num_outputs,
                      kernel_size=kernel_size,
                      activation_fn=None,
                      stride=stride,
                      scope=scope
                    )
    return net


def __deconv_bn_relu(net, kernel_size, stride, num_outputs, scope='deconv'):
    net = slim.conv2d_transpose(net,
                                kernel_size=kernel_size,
                                stride=stride,
                                num_outputs=num_outputs,
                                activation_fn=None,
                                scope=scope)
    with tf.variable_scope(scope):
        net = slim.batch_norm(net, scope='bn')
        net = tf.nn.relu(net, name='relu')
        return net


def __bn_relu_conv(net, kernel_size, stride, num_outputs, scope=''):
    with tf.variable_scope(scope):
        net = slim.batch_norm(net, scope='bn')
        net = tf.nn.relu(net, name='relu')
        net = __conv(net, kernel_size, stride, num_outputs)
        return net


def __conv_bn_relu(net, kernel_size, stride, num_outputs, scope=''):
    with tf.variable_scope(scope):
        net = __conv(net, kernel_size, stride, num_outputs)
        net = slim.batch_norm(net, scope='bn')
        net = tf.nn.relu(net, name='relu')
        return net


def __bn_crelu(net):
    net = slim.batch_norm(net, scope='bn')
    # negation of bn results
    with tf.name_scope('neg'):
        neg_net = -net
    # concat bn and neg-bn
    with tf.name_scope('concat'):
        net = tf.concat([net, neg_net], axis=-1)
    # relu
    net = tf.nn.relu(net, name='relu')
    return net


def __conv_bn_crelu(net, kernel_size, stride, num_outputs, scope=''):
    with tf.variable_scope(scope):
        net = __conv(net, kernel_size, stride, num_outputs)
        return __bn_crelu(net)


def __bn_crelu_conv(net, kernel_size, stride, num_outputs, scope=''):
    with tf.variable_scope(scope):
        net = __bn_crelu(net)
        return __conv(net, kernel_size, stride, num_outputs)


def __mCReLU(inputs, mc_config):
    """
    every cReLU has at least three conv steps:
        conv_bn_relu, conv_bn_crelu, conv_bn_relu
    if the inputs has a different number of channels as crelu output,
    an extra 1x1 conv is added before sum.
    """
    if mc_config.preact_bn:
        conv1_fn = __bn_relu_conv
        conv1_scope = '1'
    else:
        conv1_fn = __conv
        conv1_scope = '1/conv'

    sub_conv1 = conv1_fn(inputs,
                        kernel_size=1,
                        stride=mc_config.stride,
                        num_outputs=mc_config.num_outputs[0],
                        scope=conv1_scope)

    sub_conv2 = __bn_relu_conv(sub_conv1,
                            kernel_size=3,
                            stride=1,
                            num_outputs=mc_config.num_outputs[1],
                            scope='2')

    sub_conv3 = __bn_crelu_conv(sub_conv2,
                                kernel_size=1,
                                stride=1,
                                num_outputs=mc_config.num_outputs[2],
                                scope='3')

    if inputs.shape.as_list()[-1] == mc_config.num_outputs[2]:
        conv_proj = inputs
    else:
        if mc_config.proj_type == 'conv':
            conv_proj = __conv(inputs,
                            kernel_size=1,
                            stride=mc_config.stride,
                            num_outputs=mc_config.num_outputs[2],
                            scope='proj')
        elif mc_config.proj_type == 'zero':
            num_inputs = inputs.shape.as_list()[-1]
            num_outputs = mc_config.num_outputs[2]
            conv_proj = slim.pool(inputs,
                                 kernel_size=(mc_config.stride, mc_config.stride),
                                 stride=mc_config.stride,
                                 pooling_type='AVG',
                                 scope='proj_pool')
            conv_proj = tf.pad(conv_proj, [[0, 0], [0, 0], [0, 0],
                                          [(num_outputs-num_inputs)//2, (num_outputs-num_inputs)//2]])
        else:
            raise ValueError('Projection type %s not supported'%mc_config.proj_type)

    conv = sub_conv3 + conv_proj
    return conv


def __inception_block(inputs, block_config):
    num_outputs = block_config.num_outputs.split()  # e.g. 64 24-48-48 128
    stride = block_config.stride
    num_outputs = [s.split('-') for s in num_outputs]
    inception_outputs = int(num_outputs[-1][0])
    num_outputs = num_outputs[:-1]
    pool_path_outputs = None
    if stride > 1:
        pool_path_outputs = num_outputs[-1][0]
        num_outputs = num_outputs[:-1]

    scopes = [['0']]  # follow the name style of caffe pva
    kernel_sizes = [[1]]
    for path_idx, path_outputs in enumerate(num_outputs[1:]):
        path_idx += 1
        path_scopes = ['{}_reduce'.format(path_idx)]
        path_scopes.extend(['{}_{}'.format(path_idx, i - 1)
                            for i in range(1, len(path_outputs))])
        scopes.append(path_scopes)

        path_kernel_sizes = [1, 3, 3][:len(path_outputs)]
        kernel_sizes.append(path_kernel_sizes)

    paths = []
    if block_config.preact_bn:
        preact = slim.batch_norm(inputs, scope='bn')
        preact = tf.nn.relu(preact, name='relu')
    else:
        preact = inputs

    path_params = zip(num_outputs, scopes, kernel_sizes)
    for path_idx, path_param in enumerate(path_params):
        path_net = preact
        for conv_idx, (num_output, scope, kernel_size) in \
                enumerate(zip(*path_param)):
            if conv_idx == 0:
                conv_stride = stride
            else:
                conv_stride = 1
            path_net = __conv_bn_relu(path_net, kernel_size,
                                    conv_stride, num_output, scope)
        paths.append(path_net)

    if stride > 1:
        path_net = slim.pool(inputs, kernel_size=3, padding='SAME',
                            stride=2, scope='pool')
        path_net = __conv_bn_relu(path_net,
                                kernel_size=1,
                                stride=1,
                                num_outputs=pool_path_outputs,
                                scope='poolproj')
        paths.append(path_net)
    block_net = tf.concat(paths, axis=-1)
    block_net = __conv(block_net,
                    kernel_size=1,
                    stride=1,
                    num_outputs=inception_outputs,
                    scope='out/conv')

    if inputs.shape.as_list()[-1] == inception_outputs:
        proj = inputs
    else:
        if block_config.proj_type == 'conv':
            proj = __conv(inputs,
                        kernel_size=1,
                        stride=stride,
                        num_outputs=inception_outputs,
                        scope='proj')
        elif block_config.proj_type == 'zero':
            num_inputs = inputs.shape.as_list()[-1]
            proj = slim.pool(inputs,
                            kernel_size=(stride, stride),
                            stride=stride,
                            pooling_type='AVG',
                            scope='proj_pool')
            proj = tf.pad(proj, [[0, 0], [0, 0], [0, 0],
                                          [(inception_outputs-num_inputs)//2, (inception_outputs-num_inputs)//2]])
        else:
            raise ValueError('Projection type %s not supported'%block_config.proj_type)

    return block_net + proj


def __conv_stage(inputs, block_configs, scope, end_points):
    net = inputs
    for idx, bc in enumerate(block_configs):
        if bc.block_type == BLOCK_TYPE_MCRELU:
            block_scope = '{}_{}'.format(scope, idx + 1)
            fn = __mCReLU
        elif bc.block_type == BLOCK_TYPE_INCEP:
            block_scope = '{}_{}/incep'.format(scope, idx + 1)
            fn = __inception_block
        with tf.variable_scope(block_scope):
            net = fn(net, bc)
            end_points[block_scope] = net
    end_points[scope] = net
    return net


def pvanet_scope(is_training,
                weights_initializer=slim.xavier_initializer(),
                batch_norm_param_initializer=None,
                weight_decay=0.99):
    l2_regularizer = slim.l2_regularizer(weight_decay)
    with slim.arg_scope([slim.conv2d],
                        padding='SAME',
                        weights_initializer=weights_initializer,
                        weights_regularizer=l2_regularizer,
                        trainable=is_training
                        ):
        with slim.arg_scope([slim.batch_norm],
                            is_training=is_training,
                            decay=0.9,
                            scale=True,
                            center=True,
                            param_initializers=batch_norm_param_initializer
                            ):
            with slim.arg_scope([slim.pool],
                                pooling_type='MAX',
                                padding='SAME'):
                with slim.arg_scope([slim.fully_connected],
                                    trainable=is_training,
                                    activation_fn=None,
                                    weights_regularizer=l2_regularizer) as sc:
                    return sc


def pvanet(net, num_classes,
           include_last_bn_relu=True,
           use_concat=True,
           fatness=2,
           proj_type='zero'):
    print('pvanet fatness is %d'%fatness)
    batch_size = tf.cast(net.get_shape()[0], tf.int32)
    end_points = {}

    # conv stage 1
    conv1_1 = __conv_bn_crelu(net,
                            kernel_size=(5, 5),
                            stride=1,
                            num_outputs=8*fatness,
                            scope='conv1_1')

    # pooling stage 1
    pool1 = slim.pool(conv1_1, kernel_size=3, stride=2, scope='pool1')

    # conv stage 2
    conv2_setting = (12*fatness, 12*fatness, 32*fatness)
    conv2 = __conv_stage(pool1,
                        block_configs=[
                            BlockConfig(2, conv2_setting, False, BLOCK_TYPE_MCRELU, proj_type),
                            BlockConfig(1, conv2_setting, True, BLOCK_TYPE_MCRELU, proj_type),
                            BlockConfig(1, conv2_setting, True, BLOCK_TYPE_MCRELU, proj_type)],
                        scope='conv2',
                        end_points=end_points)

    # conv stage 3
    conv3_setting = (24*fatness, 24*fatness, 64*fatness)
    conv3 = __conv_stage(conv2,
                        block_configs=[
                            BlockConfig(2, conv3_setting, True, BLOCK_TYPE_MCRELU, proj_type),
                            BlockConfig(1, conv3_setting, True, BLOCK_TYPE_MCRELU, proj_type),
                            BlockConfig(1, conv3_setting, True, BLOCK_TYPE_MCRELU, proj_type),
                            BlockConfig(1, conv3_setting, True, BLOCK_TYPE_MCRELU, proj_type)],
                        scope='conv3',
                        end_points=end_points)

    def str_join(lst, separator=' '):
        return separator.join(lst)

    # conv stage 4
    conv4_pool_setting = str_join([str(32 * fatness),
                                   str_join([str(24 * fatness), str(64 * fatness)], '-'),
                                   str_join([str(12 * fatness), str(24 * fatness), str(24 * fatness)], '-'),
                                   str(64 * fatness),
                                   str(128 * fatness)])
    conv4_path_setting = str_join([str(32 * fatness),
                                   str_join([str(32 * fatness), str(64 * fatness)], '-'),
                                   str_join([str(12 * fatness), str(24 * fatness), str(24 * fatness)], '-'),
                                   str(128 * fatness)])
    conv4 = __conv_stage(conv3,
                        block_configs=[
                            BlockConfig(2, conv4_pool_setting, True, BLOCK_TYPE_INCEP, proj_type),
                            BlockConfig(1, conv4_path_setting, True, BLOCK_TYPE_INCEP, proj_type),
                            BlockConfig(1, conv4_path_setting, True, BLOCK_TYPE_INCEP, proj_type),
                            BlockConfig(1, conv4_path_setting, True, BLOCK_TYPE_INCEP, proj_type)],
                        scope='conv4',
                        end_points=end_points)

    # conv stage 5
    conv5_pool_setting = str_join([str(32 * fatness),
                                   str_join([str(48 * fatness), str(96 * fatness)], '-'),
                                   str_join([str(16 * fatness), str(32 * fatness), str(32 * fatness)], '-'),
                                   str(64 * fatness),
                                   str(192 * fatness)])
    conv5_path_setting = str_join([str(32 * fatness),
                                   str_join([str(48 * fatness), str(96 * fatness)], '-'),
                                   str_join([str(16 * fatness), str(32 * fatness), str(32 * fatness)], '-'),
                                   str(192 * fatness)])
    conv5 = __conv_stage(conv4,
                        block_configs=[
                            BlockConfig(2, conv5_pool_setting, True, BLOCK_TYPE_INCEP, proj_type),
                            BlockConfig(1, conv5_path_setting, True, BLOCK_TYPE_INCEP, proj_type),
                            BlockConfig(1, conv5_path_setting, True, BLOCK_TYPE_INCEP, proj_type),
                            BlockConfig(1, conv5_path_setting, True, BLOCK_TYPE_INCEP, proj_type)],
                        scope='conv5',
                        end_points=end_points)

    if include_last_bn_relu:
        with tf.variable_scope('conv5_4'):
            last_bn = slim.batch_norm(conv5, scope='last_bn')
            conv5 = tf.nn.relu(last_bn)
    end_points['conv5'] = conv5

    if use_concat:
        down = slim.pool(conv3, kernel_size=3, stride=2, scope='down/conv')
        end_points['down/conv'] = down

        up = __deconv_bn_relu(conv5,
                            num_outputs=conv5.get_shape()[3],
                            kernel_size=(4, 4),
                            stride=2,
                            scope='up')
        end_points['up'] = up

        with tf.variable_scope('concat'):
            concat = tf.concat([down, conv4, up], axis=-1)
            end_points['concat'] = concat
        convf = __conv_bn_relu(concat, kernel_size=(1, 1), stride=1, num_outputs=concat.get_shape()[-1], scope='convf')
        # use fc
        net = tf.reduce_mean(convf, axis=[1, 2])
    else:
        # use fc
        net = tf.reduce_mean(conv5, axis=[1, 2])

    net = tf.reshape(net, [batch_size, -1])
    net = slim.fully_connected(net,
                            num_outputs=num_classes,
                            scope='fc')
    end_points['fc'] = net

    return net, end_points




