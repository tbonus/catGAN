import os
import tensorflow as tf


def prepare_data(config):
    height, width, channels = config['height'], config['width'], config['channels']
    batch_size = config['batch_size']

    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, config['data_dir'])

    images = []

    for each in os.listdir(data_dir):
        if 'cat' in each:
            images.append(os.path.join(data_dir, each))

    all_images = tf.convert_to_tensor(images, dtype=tf.string)

    images_queue = tf.train.slice_input_producer([all_images])

    content = tf.read_file(images_queue[0])
    image = tf.image.decode_jpeg(content, channels=channels)

    image = tf.image.resize_images(image, size=[height, width])
    image.set_shape([height, width, channels])

    # convert images to floats
    tf.cast(image, tf.float32)

    # reduce images value range to (0, 1) from (0, 255)
    image = image / 255.0

    images_batch = tf.train.shuffle_batch([image],
                                          batch_size=batch_size,
                                          capacity=200 + 3 * batch_size,
                                          min_after_dequeue=200,
                                          num_threads=4)

    num_images = len(images)

    return images_batch, num_images


if __name__ == '__main__':
    prepare_data()
