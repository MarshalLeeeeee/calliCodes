import tensorflow as tf
import numpy as np
import argparse
import os
from util import locate, choice, find_truth, loader, plot, save_vector
from network import cycle_consistent_vae_with_gan

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help="gpu to use")
    parser.add_argument('--gpu_fraction', type=float, default=0.8, help="fraction of gpu memory to use")
    parser.add_argument('--categorical_cardinality', type=int, default=100, help="number of the characters to be loaded")
    parser.add_argument('--data_path', type=str, default='../../demo/', help="path to save images")
    parser.add_argument('--styles', type=str, default='klx', help="calligraphy style (sub folders)")
    parser.add_argument('--image_size', type=int, default=64, help="the size of images trained")
    parser.add_argument('--force_grayscale', type=bool, default=True, help="transform images into single channel or not")
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('--lr', type=float, default=5e-4, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--epochs', type=int, default=1000, help="epochs")
    parser.add_argument('--kernel', type=int, default=4, help="kernel size")
    parser.add_argument('--stride', type=int, default=2, help="stride")
    parser.add_argument('--class_dim', type=int, default=30, help="dimension of class vector")
    parser.add_argument('--reconstruct_coef', type=float, default=1.0, help="reconstruct coef")
    parser.add_argument('--generator_coef', type=float, default=1.0, help="generator coef")
    parser.add_argument('--discriminator_coef', type=float, default=1.0, help="discriminator coef")
    return parser.parse_args()

def test_with_graph_manually_set_up():
    parser = init()
    os.environ["CUDA_VISIBLE_DEVICES"] = parser.gpu
    categorical_cardinality = parser.categorical_cardinality
    data_path = parser.data_path
    styles = parser.styles
    image_size = parser.image_size
    force_grayscale = parser.force_grayscale
    channel_size = 1 if force_grayscale else 3
    seed = parser.seed
    lr = parser.lr
    batch_size = parser.batch_size
    epochs = parser.epochs
    kernel = parser.kernel
    stride = parser.stride
    class_dim = parser.class_dim
    reconstruct_coef = parser.reconstruct_coef
    generator_coef = parser.generator_coef
    discriminator_coef = parser.discriminator_coef

    imageName, imageDict = locate(data_path, styles=styles, max_label=categorical_cardinality)
    _, imageTrue = locate(data_path, max_label=categorical_cardinality)
    imageNum = len(imageName)

    image1 = tf.placeholder(tf.float32,[None, image_size, image_size, channel_size],name="image1")
    image2 = tf.placeholder(tf.float32,[None, image_size, image_size, channel_size],name="image2")
    is_training = tf.placeholder(tf.bool,[],name="is_training")

    forward_loss, reconstruct_loss, generator_loss, discriminator_loss, image1_forward_reconstruct, vector = cycle_consistent_vae_with_gan(
                                                                                                             image1,image2,kernel,stride,class_dim,is_training,
                                                                                                             reconstruct_coef,generator_coef,discriminator_coef,
                                                                                                             'cycle-consistent-vae-with-gan')

    config = tf.ConfigProto() 
    config.gpu_options.per_process_gpu_memory_fraction = parser.gpu_fraction
    saver = tf.train.Saver()
    idxes_1 = np.arange(imageNum, dtype=np.int32)
    np.random.shuffle(idxes_1)
    with tf.Session(config=config) as sess:
        saver.restore(sess, tf.train.latest_checkpoint('ckpt/server-5'))
        for idx in range(0, imageNum, batch_size):
            image1_test = loader(imageName[idxes_1[idx:idx + batch_size]],desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),force_grayscale=force_grayscale)
            image2_test = loader(find_truth(imageName[idxes_1[idx:idx + batch_size]],imageTrue),desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),force_grayscale=force_grayscale)
            feed_dict_not_training = {image1:image1_test,image2:image2_test,is_training:False}
            image_reconstruct, latent_vector = sess.run([image1_forward_reconstruct, vector],feed_dict=feed_dict_not_training)
            save_vector(imageName[idxes_1[idx:idx + batch_size]], latent_vector)
            #plot(image1_test, image_reconstruct, image2_test, 0)

def test_with_graph_automatically_loaded():
    parser = init()
    os.environ["CUDA_VISIBLE_DEVICES"] = parser.gpu
    categorical_cardinality = parser.categorical_cardinality
    data_path = parser.data_path
    styles = parser.styles
    image_size = parser.image_size
    force_grayscale = parser.force_grayscale
    channel_size = 1 if force_grayscale else 3
    seed = parser.seed
    lr = parser.lr
    batch_size = parser.batch_size
    epochs = parser.epochs
    kernel = parser.kernel
    stride = parser.stride
    class_dim = parser.class_dim
    reconstruct_coef = parser.reconstruct_coef
    generator_coef = parser.generator_coef
    discriminator_coef = parser.discriminator_coef

    imageName, imageDict = locate(data_path, styles=styles, max_label=categorical_cardinality)
    _, imageTrue = locate(data_path, max_label=categorical_cardinality)
    imageNum = len(imageName)

    #image1 = tf.placeholder(tf.float32,[None, image_size, image_size, channel_size],name="image1")
    #image2 = tf.placeholder(tf.float32,[None, image_size, image_size, channel_size],name="image2")
    #is_training = tf.placeholder(tf.bool,[],name="is_training")

    saver = tf.train.import_meta_graph('ckpt/server-2/model.meta')
    graph = tf.get_default_graph()
    with graph.as_default():
        variable_names = [v.name for v in tf.all_variables()]
        print(variable_names)
        config = tf.ConfigProto() 
        config.gpu_options.per_process_gpu_memory_fraction = parser.gpu_fraction
        idxes_1 = np.arange(imageNum, dtype=np.int32)
        np.random.shuffle(idxes_1)
        with tf.Session(config=config) as sess:
            saver.restore(sess, tf.train.latest_checkpoint('ckpt/server-2/'))
            image1_test = loader(imageName[idxes_1[0:10]],desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),force_grayscale=force_grayscale)
            image2_test = loader(find_truth(imageName[idxes_1[0:10]],imageTrue),desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),force_grayscale=force_grayscale)
            image1 = graph.get_tensor_by_name('image1:0')
            image2 = graph.get_tensor_by_name('image2:0')
            is_training = graph.get_tensor_by_name('is_training:0')
            feed_dict_not_training = {image1:image1_test,image2:image2_test,is_training:False}
            class_vector = graph.get_operation_by_name('cycle-consistent-vae-with-gan/encoder/class_vector/fully_connected/Maximum')
            image_reconstruct = graph.get_operation_by_name('cycle-consistent-vae-with-gan/decoder/conv4/Conv2d_transpose/Sigmoid')
            _class_vector, _image_reconstruct = sess.run([class_vector, image_reconstruct],feed_dict = feed_dict_not_training)
            print(_class_vector.shape)
            plot(image1_test, _image_reconstruct, image2_test, 0)


if __name__ == '__main__':
    test_with_graph_manually_set_up()
    #test_with_graph_automatically_loaded()