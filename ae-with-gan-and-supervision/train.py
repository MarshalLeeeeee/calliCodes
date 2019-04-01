import tensorflow as tf
import numpy as np
import argparse
import os
from util import locate, choice, find_truth, loader, plot
from network import cycle_consistent_vae_with_gan, scope_variables, get_mean

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help="gpu to use")
    parser.add_argument('--gpu_fraction', type=float, default=0.8, help="fraction of gpu memory to use")
    parser.add_argument('--categorical_cardinality', type=int, default=100, help="number of the characters to be loaded")
    parser.add_argument('--data_path', type=str, default='../../demo/', help="path to save images")
    parser.add_argument('--styles', type=str, default='cklxz', help="calligraphy style (sub folders)")
    parser.add_argument('--image_size', type=int, default=64, help="the size of images trained")
    parser.add_argument('--force_grayscale', type=bool, default=True, help="transform images into single channel or not")
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--epochs', type=int, default=1000, help="epochs")
    parser.add_argument('--kernel', type=int, default=4, help="kernel size")
    parser.add_argument('--stride', type=int, default=2, help="stride")
    parser.add_argument('--class_dim', type=int, default=75, help="dimension of class vector")
    parser.add_argument('--reconstruct_coef', type=float, default=1.0, help="reconstruct coef")
    parser.add_argument('--generator_coef', type=float, default=1.0, help="generator coef")
    parser.add_argument('--discriminator_coef', type=float, default=1.0, help="discriminator coef")
    return parser.parse_args()

def main():
    # initialize parameters
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

    # load data
    imageName, imageDict = locate(data_path, styles=styles, max_label=categorical_cardinality)
    _, imageTrue = locate(data_path, max_label=categorical_cardinality)
    imageNum = len(imageName)

    image1 = tf.placeholder(tf.float32,[None, image_size, image_size, channel_size],name="image1")
    image2 = tf.placeholder(tf.float32,[None, image_size, image_size, channel_size],name="image2")
    is_training = tf.placeholder(tf.bool,[],name="is_training")

    forward_loss, reconstruct_loss, generator_loss, discriminator_loss, image1_forward_reconstruct, _ = cycle_consistent_vae_with_gan(
                                                                                                        image1,image2,kernel,stride,class_dim,is_training,
                                                                                                        reconstruct_coef,generator_coef,discriminator_coef,
                                                                                                        'cycle-consistent-vae-with-gan')

    encoder_variables = scope_variables("cycle-consistent-vae-with-gan/encoder")
    decoder_variables = scope_variables('cycle-consistent-vae-with-gan/decoder')
    discriminator_variables = scope_variables('cycle-consistent-vae-with-gan/discriminator')

    forward_solver = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.5)
    generator_solver = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.5)
    discriminator_solver = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.5)
    forward_train = forward_solver.minimize(forward_loss, var_list=encoder_variables+decoder_variables)
    generator_train = generator_solver.minimize(generator_loss, var_list=decoder_variables)
    discriminator_train = discriminator_solver.minimize(discriminator_loss, var_list=discriminator_variables)

    idxes_1 = np.arange(imageNum, dtype=np.int32)
    config = tf.ConfigProto() 
    config.gpu_options.per_process_gpu_memory_fraction = parser.gpu_fraction
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            np.random.shuffle(idxes_1)
            forward_losses = []
            generator_losses = []
            discriminator_losses = []
            
            for idx in range(0, imageNum, batch_size):
                image1_batch = loader(imageName[idxes_1[idx:idx + batch_size]],desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),force_grayscale=force_grayscale)
                image2_batch = loader(find_truth(imageName[idxes_1[idx:idx + batch_size]],imageTrue),desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),force_grayscale=force_grayscale)

                feed_dict_training = {image1:image1_batch,image2:image2_batch,is_training:True}

                # forward
                _,_forward_loss = sess.run([forward_train,forward_loss],feed_dict=feed_dict_training)
                forward_losses.append(_forward_loss)

                # generator
                _,_generator_loss = sess.run([generator_train,generator_loss],feed_dict=feed_dict_training)
                generator_losses.append(_generator_loss)

                # discriminator
                _,_discriminator_loss = sess.run([discriminator_train,discriminator_loss],feed_dict=feed_dict_training)
                discriminator_losses.append(_discriminator_loss)

            print('epoch: %d\nforward_loss: %f, generator_loss: %f, discriminator_loss: %f\n' % (epoch, get_mean(forward_losses), get_mean(generator_losses), get_mean(discriminator_losses)))
            
            image1_plot = loader(imageName[idxes_1[0:10]],desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),force_grayscale=force_grayscale)
            image2_plot = loader(find_truth(imageName[idxes_1[0:10]],imageTrue),desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),force_grayscale=force_grayscale)
            feed_dict_not_training = {image1:image1_plot,image2:image2_plot,is_training:False}
            image1_reconstruct = sess.run(image1_forward_reconstruct,feed_dict=feed_dict_not_training)
            plot(image1_plot, image1_reconstruct, image2_plot, epoch)
        saver.save(sess,'ckpt/model')


if __name__ == '__main__':
    main()
    