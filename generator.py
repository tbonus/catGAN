import tensorflow as tf


def generator(noise, config, train=True, reuse=False):
    """
    Network for generating images
    Parameters
    - noise: input tensor of shape = [BATCH_SIZE, NOISE_SIZE]
    - train: specify if network is trainable
    Returns tensor with images
    """

    # channels
    c1, c2, c3, c4, c5 = 512, 256, 128, 64, 32
    layers = 6
    size = config['width'] // (2 ** (layers - 1))
    output_dim = config['channels']

    with tf.variable_scope('generator') as scope:
        if reuse:
            scope.reuse_variables()

        w1 = tf.get_variable(name='w1',
                             shape=[config['noise_size'], size * size * c1],
                             dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable(name='b1',
                             shape=[size * size * c1],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
        conv1_flat = tf.add(tf.matmul(noise, w1), b1, name='flat_conv1')
        conv1 = tf.reshape(
            conv1_flat, shape=[-1, size, size, c1], name='conv1')
        act1 = tf.nn.relu(conv1,
                          name='activation1')

        conv2 = tf.layers.conv2d_transpose(inputs=act1,
                                           filters=c2,
                                           kernel_size=(5, 5),
                                           strides=(2, 2),
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           trainable=train,
                                           padding='same',
                                           name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2,
                                            scope='bias2')
        act2 = tf.nn.relu(bn2,
                          name='activation2')

        conv3 = tf.layers.conv2d_transpose(inputs=act2,
                                           filters=c3,
                                           kernel_size=(5, 5),
                                           strides=(2, 2),
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           trainable=train,
                                           padding='same',
                                           name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3,
                                            scope='bias3')
        act3 = tf.nn.relu(bn3,
                          name='activation3')

        conv4 = tf.layers.conv2d_transpose(inputs=act3,
                                           filters=c4,
                                           kernel_size=(5, 5),
                                           strides=(2, 2),
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           trainable=train,
                                           padding='same',
                                           name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4,
                                            scope='bias4')
        act4 = tf.nn.relu(bn4,
                          name='activation4')

        conv5 = tf.layers.conv2d_transpose(inputs=act4,
                                           filters=c5,
                                           kernel_size=(5, 5),
                                           strides=(2, 2),
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           trainable=train,
                                           padding='same',
                                           name='conv5')
        bn5 = tf.contrib.layers.batch_norm(conv5,
                                            scope='bias5')
        act5 = tf.nn.relu(bn5,
                          name='activation5')

        conv6 = tf.layers.conv2d_transpose(inputs=act5,
                                           filters=output_dim,
                                           kernel_size=(5, 5),
                                           strides=(2, 2),
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           trainable=train,
                                           padding='same',
                                           name='conv6')
        act6 = tf.nn.tanh(conv6,
                          name='activation6')

        return act6
