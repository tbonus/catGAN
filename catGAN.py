""" Neural network to generate cats and dogs images """

import tensorflow as tf
import numpy as np
from generator import generator
from discriminator import discriminator
from prepare_data import prepare_data


CONFIG = {
    'width': 128,
    'height': 128,
    'channels': 3,
    'noise_size': 100,
    'batch_size': 64,
    'epochs': 1000,
    'dis_iterations': 5,
    'gen_iterations': 1,
    'data_dir': 'celeba_cropped',
    'sigmoid': True
}

if __name__ == '__main__':
    print('Welcome to catGAN!\n')
    print('Current config:')
    for key in CONFIG:
        print('\t', key, ':', CONFIG[key])
    print('\n')

    # inputs
    with tf.variable_scope('inputs'):
        real_image = tf.placeholder(dtype=tf.float32,
                                    shape=[None,
                                           CONFIG['height'],
                                           CONFIG['width'],
                                           CONFIG['channels']],
                                    name='real_image')
        noise = tf.placeholder(dtype=tf.float32,
                               shape=[None, CONFIG['noise_size']],
                               name='noise')

    # NETWORKS

    # generator
    fake_image = generator(noise=noise, config=CONFIG)

    # discriminator
    fake_result = discriminator(image=fake_image, config=CONFIG)
    real_result = discriminator(image=real_image, config=CONFIG, reuse=True)

    # loss functions
    dis_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)
    gen_loss = -tf.reduce_mean(fake_result)

    # SUMMARY

    # add losses to summary
    dis_loss_sum = tf.summary.scalar('discriminator_loss', dis_loss)
    gen_loss_sum = tf.summary.scalar('generator_loss', gen_loss)

    # add generator images to summary
    gen_image_sum = tf.summary.image('generator_image', fake_image)

    # get variables
    variables = tf.trainable_variables()
    dis_vars = [var for var in variables if 'dis' in var.name]
    gen_vars = [var for var in variables if 'gen' in var.name]

    # trainers
    dis_trainer = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(loss=dis_loss,
                                                                         var_list=dis_vars)
    gen_trainer = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(loss=gen_loss,
                                                                         var_list=gen_vars)
    # clip weights to prevent too fast discriminator convergence
    dis_clip = [var.assign(tf.clip_by_value(var, -0.01, 0.01))
                for var in dis_vars]

    # get real images
    print('Preparing dataset')
    image_batch, image_num = prepare_data(config=CONFIG)
    batch_num = image_num // CONFIG['batch_size']
    print('Dataset ready')

    # declare model saver
    saver = tf.train.Saver()

    # create session
    with tf.Session() as sess:
        print('Setting up session:')
        # merge summary
        dis_summary = tf.summary.merge([dis_loss_sum])
        gen_summary = tf.summary.merge([gen_loss_sum, gen_image_sum])
        print('\tSummary added')

        # initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        print('\tVariables initialized')

        # save graph to tensorboard
        summary_writer = tf.summary.FileWriter('/tmp/logs/catGAN/celeba',
                                               graph=sess.graph)
        saver.save(sess=sess, save_path='model/model.ckpt')
        print('\tGraph exported to Tensorboard')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print('Learning started\n')
        global_step = 0
        for epoch in range(CONFIG['epochs']):
            # run iterations
            for i in range(batch_num):
                print('Epoch %d: Iteration %d' % (epoch, i))
                random_noise = np.random.uniform(low=-1.0,
                                                 high=1.0,
                                                 size=[CONFIG['batch_size'], CONFIG['noise_size']]).astype(np.float32)
                # train discriminator
                for d in range(CONFIG['dis_iterations']):
                    # get images
                    real_images = sess.run(image_batch)
                    sess.run(dis_clip)

                    _, dis_summary_values = sess.run([dis_trainer, dis_summary],
                                                     feed_dict={noise: random_noise, real_image: real_images})

                # train generator
                for g in range(CONFIG['gen_iterations']):
                    _, gen_summary_values = sess.run([gen_trainer, gen_summary],
                                                     feed_dict={noise: random_noise})
                # update global step
                global_step += 1

            # summary for each epoch
            summary_writer.add_summary(summary=gen_summary_values,
                                       global_step=epoch)
            summary_writer.add_summary(summary=dis_summary_values,
                                       global_step=epoch)

            if epoch % 10 == 0:
                saver.save(sess=sess,
                           save_path='model/model_%d.ckpt' % epoch)


        coord.request_stop()
        coord.join(threads)
