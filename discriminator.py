import tensorflow as tf
import numpy as np


def LReLu(x, leak=0.2, name=''):
    return tf.maximum(x, leak * x, name=name)


def discriminator(image, config, train=True, reuse=False):
    c1, c2, c3, c4 = 64, 128, 256, 512
    layers = 6
    size = config['width'] // (2 ** (layers - 1))
    output_dim = config['channels']

    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()

        conv1 = tf.layers.conv2d(inputs=image,
                                 filters=c1,
                                 kernel_size=(5, 5),
                                 strides=(2, 2),
                                 kernel_initializer=tf.truncated_normal_initializer(),
                                 padding='SAME',
                                 name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1,
                                                    scope='bias1')
        act1 = LReLu(bn1, name='act1')

        conv2 = tf.layers.conv2d(inputs=act1,
                                 filters=c2,
                                 kernel_size=(5, 5),
                                 strides=(2, 2),
                                 kernel_initializer=tf.truncated_normal_initializer(),
                                 padding='SAME',
                                 name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2,
                                                    scope='bias2')
        act2 = LReLu(bn2, name='act2')

        conv3 = tf.layers.conv2d(inputs=act2,
                                 filters=c3,
                                 kernel_size=(5, 5),
                                 strides=(2, 2),
                                 kernel_initializer=tf.truncated_normal_initializer(),
                                 padding='SAME',
                                 name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3,
                                                    scope='bias3')
        act3 = LReLu(bn3, name='act3')

        conv4 = tf.layers.conv2d(inputs=act3,
                                 filters=c4,
                                 kernel_size=(5, 5),
                                 strides=(2, 2),
                                 kernel_initializer=tf.truncated_normal_initializer(),
                                 padding='SAME',
                                 name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4,
                                                    scope='bias4')
        act4 = LReLu(bn4, name='act2')

        # calculate shape by multiplying sizes of first element
        connected_dim = int(np.prod(act4.get_shape()[1:]))
        fc1 = tf.reshape(act4, shape=[-1, connected_dim], name='fc1')

        w1 = tf.get_variable(name='w2',
                             shape=[connected_dim, 1],
                             dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable(name='b2',
                             shape=[1],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer())

        logits = tf.add(tf.matmul(fc1, w1), b1, name='logits')

        if config['sigmoid']:
            logits_sigmoid = tf.nn.sigmoid(logits, name='logits_sigmoid')
            return logits_sigmoid

        return logits
