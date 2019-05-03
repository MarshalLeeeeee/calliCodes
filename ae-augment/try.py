import tensorflow as tf
import numpy as np
import argparse
import os
from util import locate, choice, find_truth, loader, plot_batch, make_partition
from network import ae_with_gan, scope_variables, get_mean, encoder, decoder, binary

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help="gpu to use")
    parser.add_argument('--gpu_fraction', type=float, default=0.8, help="fraction of gpu memory to use")
    parser.add_argument('--categorical_cardinality', type=int, default=1000, help="number of the characters to be loaded")
    parser.add_argument('--fraction', type=float, default=0.99, help="fraction of train and test")
    parser.add_argument('--data_path', type=str, default='../../demo/', help="path to save images")
    parser.add_argument("--styles", nargs="*", type=int, default=[6,10], help="appointed styles to be trained")
    parser.add_argument('--image_size', type=int, default=64, help="the size of images trained")
    parser.add_argument('--force_grayscale', type=bool, default=True, help="transform images into single channel or not")
    parser.add_argument('--augment', type=int, default=2, help="augment level with ascending (0: None)")
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('--lr', type=float, default=5e-5, help="learning rate")
    parser.add_argument('--loss_type', type=str, default='l1', help="choice of loss functions")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--epochs', type=int, default=1000, help="epochs")
    parser.add_argument('--kernel', type=int, default=4, help="kernel size")
    parser.add_argument('--stride', type=int, default=2, help="stride")
    parser.add_argument('--class_dim', type=int, default=50, help="dimension of class vector")
    parser.add_argument('--style_dim', type=int, default=50, help="dimension of style vector")
    parser.add_argument('--reconstruct_coef_1', type=float, default=1.0, help="reconstruct coef 1")
    parser.add_argument('--reconstruct_coef_3', type=float, default=1.0, help="reconstruct coef 3")
    parser.add_argument('--generator_coef', type=float, default=1.0, help="generator coef")
    parser.add_argument('--discriminator_coef', type=float, default=1.0, help="discriminator coef")
    return parser.parse_args()

def main():
    # initialize parameters
    parser = init()
    os.environ["CUDA_VISIBLE_DEVICES"] = parser.gpu
    categorical_cardinality = parser.categorical_cardinality
    fraction = parser.fraction
    data_path = parser.data_path
    styles = parser.styles
    image_size = parser.image_size
    force_grayscale = parser.force_grayscale
    augment = parser.augment
    channel_size = 1 if force_grayscale else 3
    seed = parser.seed
    lr = parser.lr
    loss_type = parser.loss_type
    batch_size = parser.batch_size
    epochs = parser.epochs
    kernel = parser.kernel
    stride = parser.stride
    class_dim = parser.class_dim
    style_dim = parser.style_dim
    reconstruct_coef_1 = parser.reconstruct_coef_1
    reconstruct_coef_3 = parser.reconstruct_coef_3
    generator_coef = parser.generator_coef
    discriminator_coef = parser.discriminator_coef

    # load data
    partition = make_partition(300,categorical_cardinality,fraction)
    imageNameTrain, imageNameTest = locate(data_path,styles=styles, max_label=categorical_cardinality,partition=partition)
    styleNum, charNum, imageNum = imageNameTrain.shape[0], imageNameTrain.shape[1], imageNameTrain.shape[0] * imageNameTrain.shape[1]

    image1 = tf.placeholder(tf.float32,[None, image_size, image_size, channel_size],name="image1")
    image1_binary = binary(image1,image_size,channel_size,0.7)
    '''
    image1_averge_org = tf.reduce_mean(image1,[1,2,3])
    image1_average = tf.reshape(tf.reduce_mean(image1,[1,2,3]),[-1,1,1,1])
    #image1_average = tf.tile(image1_average,[1, image_size, image_size, channel_size])
    image1_average = tf.cast(tf.ones_like(image1),tf.float32)*0.7
    image1_mask = tf.cast(tf.less(image1,image1_average),tf.float32)
    image1_binary = image1*image1_mask+(1-image1_mask)
    '''

    idxes_1 = np.arange(imageNum, dtype=np.int32)
    idxes_2 = np.arange(imageNum, dtype=np.int32)
    np.random.shuffle(idxes_1)
    np.random.shuffle(idxes_2)
    config = tf.ConfigProto() 
    config.gpu_options.per_process_gpu_memory_fraction = parser.gpu_fraction
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        image1_plot,_,_,_,_,_,_ = loader(imageNameTrain,idxes_1[0:10],idxes_2[0:10],styleNum,charNum,desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),augment=augment,force_grayscale=force_grayscale)
        

        feed_dict = {image1:image1_plot}
        _image1_binary = sess.run(image1_binary,feed_dict=feed_dict)

        images = [image1_plot,_image1_binary]
        coefs = [loss_type,lr,reconstruct_coef_1,reconstruct_coef_3,generator_coef,discriminator_coef]
        plot_batch(images, 'train', 0, coefs)
        exit(0)

main()