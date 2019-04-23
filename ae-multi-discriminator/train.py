import tensorflow as tf
import numpy as np
import argparse
import os
from util import locate, choice, find_truth, loader, plot_batch
from network import ae_with_gan, scope_variables, get_mean

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help="gpu to use")
    parser.add_argument('--gpu_fraction', type=float, default=0.8, help="fraction of gpu memory to use")
    parser.add_argument('--categorical_cardinality', type=int, default=1000, help="number of the characters to be loaded")
    parser.add_argument('--fraction', type=float, default=0.99, help="fraction of train and test")
    parser.add_argument('--data_path', type=str, default='../../demo/', help="path to save images")
    parser.add_argument('--style_1', type=str, default='10', help="calligraphy style 1")
    parser.add_argument('--style_2', type=str, default='6', help="calligraphy style 2")
    parser.add_argument('--image_size', type=int, default=64, help="the size of images trained")
    parser.add_argument('--force_grayscale', type=bool, default=True, help="transform images into single channel or not")
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('--lr', type=float, default=5e-5, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--epochs', type=int, default=1000, help="epochs")
    parser.add_argument('--kernel', type=int, default=4, help="kernel size")
    parser.add_argument('--stride', type=int, default=2, help="stride")
    parser.add_argument('--class_dim', type=int, default=50, help="dimension of class vector")
    parser.add_argument('--style_dim', type=int, default=50, help="dimension of style vector")
    parser.add_argument('--reconstruct_coef_1', type=float, default=1.0, help="reconstruct coef 1")
    parser.add_argument('--reconstruct_coef_2', type=float, default=1.0, help="reconstruct coef 2")
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
    style_1 = parser.style_1
    style_2 = parser.style_2
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
    style_dim = parser.style_dim
    reconstruct_coef_1 = parser.reconstruct_coef_1
    reconstruct_coef_2 = parser.reconstruct_coef_2
    reconstruct_coef_3 = parser.reconstruct_coef_3
    generator_coef = parser.generator_coef
    discriminator_coef = parser.discriminator_coef

    # load data
    partition = np.arange(categorical_cardinality, dtype=np.int32)
    np.random.shuffle(partition)
    partition = partition[:int(categorical_cardinality*(1-fraction))]
    imageNameTrain1, imageDictTrain1, imageNameTest1, imageDictTest1 = locate(data_path, styles=['std/'+style_1+'/cut'], max_label=categorical_cardinality, partition=partition)
    imageNameTrain2, imageDictTrain2, imageNameTest2, imageDictTest2 = locate(data_path, styles=['std/'+style_2+'/cut'], max_label=categorical_cardinality, partition=partition)
    imageNameTrain3, imageDictTrain3, imageNameTest3, imageDictTest3 = locate(data_path, styles=['std/0/cut'], max_label=categorical_cardinality, partition=partition)
    imageNum = len(imageNameTrain1)

    image1 = tf.placeholder(tf.float32,[None, image_size, image_size, channel_size],name="image1")
    image2 = tf.placeholder(tf.float32,[None, image_size, image_size, channel_size],name="image2")
    image3 = tf.placeholder(tf.float32,[None, image_size, image_size, channel_size],name="image3")
    image4 = tf.placeholder(tf.float32,[None, image_size, image_size, channel_size],name="image4")
    image5 = tf.placeholder(tf.float32,[None, image_size, image_size, channel_size],name="image5")
    image6 = tf.placeholder(tf.float32,[None, image_size, image_size, channel_size],name="image6")
    is_training = tf.placeholder(tf.bool,[],name="is_training")

    forward_loss, reconstruct_loss_1, reconstruct_loss_2, reconstruct_loss_3, generator_loss, discriminator_loss, \
    image1_forward_reconstruct, image2_forward_reconstruct, image3_forward_reconstruct, image4_forward_reconstruct, \
    class_vector_1, style_vector_1, image1_style_reconstruct, image3_style_reconstruct = ae_with_gan(image1,image2,image3,image4,image5,image6,kernel,stride,class_dim,style_dim,is_training,
                                                                                                     reconstruct_coef_1,reconstruct_coef_2,reconstruct_coef_3,generator_coef,discriminator_coef,
                                                                                                     'ae-with-gan')

    encoder_variables = scope_variables("ae-with-gan/encoder")
    decoder_variables = scope_variables('ae-with-gan/decoder')
    discriminator_variables_1 = scope_variables('ae-with-gan/discriminator_1')
    discriminator_variables_2 = scope_variables('ae-with-gan/discriminator_2')
    discriminator_variables_3 = scope_variables('ae-with-gan/discriminator_3')
    #all_variables = scope_variables('ae-with-gan')
    #print([n.name for n in tf.get_default_graph().as_graph_def().node])
    #print([n.name for n in all_variables])

    forward_solver = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.5)
    generator_solver = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.5)
    discriminator_solver = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.5)
    forward_train = forward_solver.minimize(forward_loss, var_list=encoder_variables+decoder_variables)
    generator_train = generator_solver.minimize(generator_loss, var_list=decoder_variables)
    discriminator_train = discriminator_solver.minimize(discriminator_loss, var_list=discriminator_variables_1+discriminator_variables_2+discriminator_variables_3)

    idxes_1 = np.arange(imageNum, dtype=np.int32)
    idxes_2 = np.arange(imageNum, dtype=np.int32)
    config = tf.ConfigProto() 
    config.gpu_options.per_process_gpu_memory_fraction = parser.gpu_fraction
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            np.random.shuffle(idxes_1)
            np.random.shuffle(idxes_2)
            forward_losses = []
            reconstruct_losses_1 = []
            reconstruct_losses_2 = []
            reconstruct_losses_3 = []
            generator_losses = []
            discriminator_losses = []
            
            for idx in range(0, imageNum, batch_size):
                image1_batch = loader(imageNameTrain1[idxes_1[idx:idx + batch_size]],desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),force_grayscale=force_grayscale)
                image2_batch = loader(find_truth(imageNameTrain1[idxes_1[idx:idx + batch_size]],imageDictTrain3),desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),force_grayscale=force_grayscale)
                image3_batch = loader(imageNameTrain2[idxes_2[idx:idx + batch_size]],desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),force_grayscale=force_grayscale)
                image4_batch = loader(find_truth(imageNameTrain2[idxes_2[idx:idx + batch_size]],imageDictTrain3),desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),force_grayscale=force_grayscale)
                image5_batch = loader(find_truth(imageNameTrain1[idxes_1[idx:idx + batch_size]],imageDictTrain2),desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),force_grayscale=force_grayscale)
                image6_batch = loader(find_truth(imageNameTrain2[idxes_2[idx:idx + batch_size]],imageDictTrain1),desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),force_grayscale=force_grayscale)
                feed_dict_training = {image1:image1_batch,image2:image2_batch,image3:image3_batch,image4:image4_batch,image5:image5_batch,image6:image6_batch,is_training:True}

                # forward
                _,_forward_loss,_reconstruct_loss_1,_reconstruct_loss_2,_reconstruct_loss_3 = sess.run([forward_train,forward_loss,reconstruct_loss_1,reconstruct_loss_2,reconstruct_loss_3],feed_dict=feed_dict_training)
                forward_losses.append(_forward_loss)
                reconstruct_losses_1.append(_reconstruct_loss_1)
                reconstruct_losses_2.append(_reconstruct_loss_2)
                reconstruct_losses_3.append(_reconstruct_loss_3)

                # generator
                _,_generator_loss = sess.run([generator_train,generator_loss],feed_dict=feed_dict_training)
                generator_losses.append(_generator_loss)

                # discriminator
                _,_discriminator_loss = sess.run([discriminator_train,discriminator_loss],feed_dict=feed_dict_training)
                discriminator_losses.append(_discriminator_loss)

            print('epoch: %d\nforward_loss: %f\nself_reconstruct_loss: %f\ntruth_reconstruct_loss: %f\ntransfer_reconstruct_loss: %f\ngenerator_loss: %f\ndiscriminator_loss: %f\n' % \
                (epoch, get_mean(forward_losses), get_mean(reconstruct_losses_1), get_mean(reconstruct_losses_2), get_mean(reconstruct_losses_3), get_mean(generator_losses), get_mean(discriminator_losses)))
            
            # test
            image1_plot = loader(imageNameTrain1[idxes_1[0:10]],desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),force_grayscale=force_grayscale)
            image2_plot = loader(find_truth(imageNameTrain1[idxes_1[0:10]],imageDictTrain3),desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),force_grayscale=force_grayscale)
            image3_plot = loader(imageNameTrain2[idxes_2[0:10]],desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),force_grayscale=force_grayscale)
            image4_plot = loader(find_truth(imageNameTrain2[idxes_2[0:10]],imageDictTrain3),desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),force_grayscale=force_grayscale)
            image5_plot = loader(find_truth(imageNameTrain1[idxes_1[0:10]],imageDictTrain2),desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),force_grayscale=force_grayscale)
            image6_plot = loader(find_truth(imageNameTrain2[idxes_2[0:10]],imageDictTrain1),desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),force_grayscale=force_grayscale)
            feed_dict_not_training = {image1:image1_plot,image2:image2_plot,image3:image3_plot,image4:image4_plot,image5:image5_plot,image6:image6_plot,is_training:False}
            _image1_forward_reconstruct,_image2_forward_reconstruct,_image3_forward_reconstruct,_image4_forward_reconstruct,_image1_style_reconstruct,_image3_style_reconstruct = sess.run([image1_forward_reconstruct,image2_forward_reconstruct,image3_forward_reconstruct,image4_forward_reconstruct,image1_style_reconstruct,image3_style_reconstruct],feed_dict=feed_dict_not_training)
            images = [image1_plot,image2_plot,image3_plot,image4_plot,image5_plot,image6_plot,_image1_forward_reconstruct,_image2_forward_reconstruct,_image3_forward_reconstruct,_image4_forward_reconstruct,_image1_style_reconstruct,_image3_style_reconstruct]
            coefs = [reconstruct_coef_1,reconstruct_coef_2,reconstruct_coef_3]
            plot_batch(images, 'train', epoch, coefs)

            image1_plot = loader(imageNameTest1,desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),force_grayscale=force_grayscale)
            image2_plot = loader(find_truth(imageNameTest1,imageDictTest3),desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),force_grayscale=force_grayscale)
            image3_plot = loader(imageNameTest2,desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),force_grayscale=force_grayscale)
            image4_plot = loader(find_truth(imageNameTest2,imageDictTest3),desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),force_grayscale=force_grayscale)
            image5_plot = loader(find_truth(imageNameTest1,imageDictTest2),desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),force_grayscale=force_grayscale)
            image6_plot = loader(find_truth(imageNameTest2,imageDictTest1),desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),force_grayscale=force_grayscale)
            feed_dict_not_training = {image1:image1_plot,image2:image2_plot,image3:image3_plot,image4:image4_plot,image5:image5_plot,image6:image6_plot,is_training:False}
            _image1_forward_reconstruct,_image2_forward_reconstruct,_image3_forward_reconstruct,_image4_forward_reconstruct,_image1_style_reconstruct,_image3_style_reconstruct = sess.run([image1_forward_reconstruct,image2_forward_reconstruct,image3_forward_reconstruct,image4_forward_reconstruct,image1_style_reconstruct,image3_style_reconstruct],feed_dict=feed_dict_not_training)
            images = [image1_plot,image2_plot,image3_plot,image4_plot,image5_plot,image6_plot,_image1_forward_reconstruct,_image2_forward_reconstruct,_image3_forward_reconstruct,_image4_forward_reconstruct,_image1_style_reconstruct,_image3_style_reconstruct]
            coefs = [reconstruct_coef_1,reconstruct_coef_2,reconstruct_coef_3]
            plot_batch(images, 'test', epoch, coefs)

        saver.save(sess,os.path.join(os.path.join('ckpt',str(reconstruct_coef_1)+'-'+str(reconstruct_coef_2)+'-'+str(reconstruct_coef_3)),'model'))


if __name__ == '__main__':
    main()
    