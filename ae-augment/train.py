import tensorflow as tf
import numpy as np
import argparse
import os
from util import locate, choice, find_truth, loader, plot_batch, make_partition
from network import ae_with_gan, scope_variables, get_mean

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help="gpu to use")
    parser.add_argument('--gpu_fraction', type=float, default=0.88, help="fraction of gpu memory to use")
    parser.add_argument('--save_frequency', type=int, default=100, help="frequency to save the model")
    parser.add_argument('--categorical_cardinality', type=int, default=1000, help="number of the characters to be loaded")
    parser.add_argument('--fraction', type=float, default=0.99, help="fraction of train and test")
    parser.add_argument('--data_path', type=str, default='../../demo/', help="path to save images")
    parser.add_argument("--styles", nargs="*", type=int, default=[6,10], help="appointed styles to be trained")
    parser.add_argument('--image_size', type=int, default=64, help="the size of images trained")
    parser.add_argument('--force_grayscale', type=bool, default=True, help="transform images into single channel or not")
    parser.add_argument('--augment', type=int, default=2, help="augment level with ascending (0: None)")
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('--lr', type=float, default=5e-5, help="learning rate")
    parser.add_argument('--loss_type', type=str, default='ce', help="choice of loss functions")
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
    save_frequency = parser.save_frequency
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
    #partition = np.array([297,304,313,316,376,381,441,512,617,633])
    imageNameTrain, imageNameTest = locate(data_path,styles=styles, max_label=categorical_cardinality,partition=partition)
    styleTrainNum, charTrainNum, imageTrainNum = imageNameTrain.shape[0], imageNameTrain.shape[1], imageNameTrain.shape[0] * imageNameTrain.shape[1]
    styleTestNum, charTestNum, imageTestNum = imageNameTest.shape[0], imageNameTest.shape[1], imageNameTest.shape[0] * imageNameTest.shape[1]
    print('partition:\n',partition)
    print(imageNameTrain.shape, imageNameTest.shape)

    image1 = tf.placeholder(tf.float32,[None, image_size, image_size, channel_size],name="image1")
    image3 = tf.placeholder(tf.float32,[None, image_size, image_size, channel_size],name="image3")
    image5 = tf.placeholder(tf.float32,[None, image_size, image_size, channel_size],name="image5")
    image6 = tf.placeholder(tf.float32,[None, image_size, image_size, channel_size],name="image6")
    label1 = tf.placeholder(tf.float32,[None, styleTrainNum],name="label1")
    label3 = tf.placeholder(tf.float32,[None, styleTrainNum],name="label3")
    is_calligraphy = tf.placeholder(tf.float32,[None, 1],name="is_calligraphy")
    is_training = tf.placeholder(tf.bool,[],name="is_training")

    forward_loss, reconstruct_loss_1, reconstruct_loss_3, generator_loss, discriminator_loss, \
    image1_forward_reconstruct, image3_forward_reconstruct, image1_style_reconstruct, image3_style_reconstruct, \
    _,_,_,_,_,_,_,_,_,_,_,_ = ae_with_gan(image1,image3,image5,image6,label1,label3,is_calligraphy,kernel,stride,class_dim,style_dim,image_size,channel_size,is_training, 
                                          loss_type,styleTrainNum,reconstruct_coef_1,reconstruct_coef_3,generator_coef,discriminator_coef,
                                          'ae-with-gan')

    encoder_variables = scope_variables("ae-with-gan/encoder")
    decoder_variables = scope_variables('ae-with-gan/decoder')
    discriminator_variables = scope_variables('ae-with-gan/discriminator')
    #all_variables = scope_variables('ae-with-gan')
    #print([n.name for n in tf.get_default_graph().as_graph_def().node])
    #print([n.name for n in all_variables])

    forward_solver = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.5)
    generator_solver = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.5)
    discriminator_solver = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.5)
    forward_train = forward_solver.minimize(forward_loss, var_list=encoder_variables+decoder_variables)
    generator_train = generator_solver.minimize(generator_loss, var_list=decoder_variables)
    discriminator_train = discriminator_solver.minimize(discriminator_loss, var_list=discriminator_variables)

    idxesTrain_1 = np.arange(imageTrainNum, dtype=np.int32)
    idxesTrain_2 = np.arange(imageTrainNum, dtype=np.int32)
    idxesTest_1 = np.arange(imageTestNum, dtype=np.int32)
    idxesTest_2 = np.arange(imageTestNum, dtype=np.int32)
    config = tf.ConfigProto() 
    config.gpu_options.per_process_gpu_memory_fraction = parser.gpu_fraction
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            np.random.shuffle(idxesTrain_1)
            np.random.shuffle(idxesTrain_2)
            forward_losses = []
            reconstruct_losses_1 = []
            reconstruct_losses_3 = []
            generator_losses = []
            discriminator_losses = []
            
            for idx in range(0, imageTrainNum, batch_size):
                image1_batch, image3_batch, image5_batch, image6_batch, label1_batch, label3_batch, is_calligraphy_batch = loader(imageNameTrain,idxesTrain_1[idx:idx + batch_size],idxesTrain_2[idx:idx + batch_size],styleTrainNum,charTrainNum,desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),augment=augment,force_grayscale=force_grayscale)
                feed_dict_training = {image1:image1_batch,image3:image3_batch,image5:image5_batch,image6:image6_batch,label1:label1_batch,label3:label3_batch,is_calligraphy:is_calligraphy_batch,is_training:True}

                # forward
                _,_forward_loss,_reconstruct_loss_1,_reconstruct_loss_3 = sess.run([forward_train,forward_loss,reconstruct_loss_1,reconstruct_loss_3],feed_dict=feed_dict_training)
                forward_losses.append(_forward_loss)
                reconstruct_losses_1.append(_reconstruct_loss_1)
                reconstruct_losses_3.append(_reconstruct_loss_3)

                # generator
                _,_generator_loss = sess.run([generator_train,generator_loss],feed_dict=feed_dict_training)
                generator_losses.append(_generator_loss)

                # discriminator
                _,_discriminator_loss = sess.run([discriminator_train,discriminator_loss],feed_dict=feed_dict_training)
                discriminator_losses.append(_discriminator_loss)

            print('epoch: %d\nforward_loss: %f\nself_reconstruct_loss: %f\ntransfer_reconstruct_loss: %f\ngenerator_loss: %f\ndiscriminator_loss: %f\n' % \
                (epoch, get_mean(forward_losses), get_mean(reconstruct_losses_1), get_mean(reconstruct_losses_3), get_mean(generator_losses), get_mean(discriminator_losses)))
            
            # test
            image1_plot, image3_plot, image5_plot, image6_plot, label1_plot, label3_plot, is_calligraphy_plot = loader(imageNameTrain,idxesTrain_1[0:10],idxesTrain_2[0:10],styleTrainNum,charTrainNum,desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),augment=augment,force_grayscale=force_grayscale)
            feed_dict_not_training = {image1:image1_plot,image3:image3_plot,image5:image5_plot,image6:image6_plot,label1:label1_plot,label3:label3_plot,is_calligraphy:is_calligraphy_plot,is_training:False}
            _image1_forward_reconstruct,_image3_forward_reconstruct,_image1_style_reconstruct,_image3_style_reconstruct = sess.run([image1_forward_reconstruct,image3_forward_reconstruct,image1_style_reconstruct,image3_style_reconstruct],feed_dict=feed_dict_not_training)
            images = [image1_plot,image3_plot,image5_plot,image6_plot,_image1_forward_reconstruct,_image3_forward_reconstruct,_image1_style_reconstruct,_image3_style_reconstruct]
            coefs = [loss_type,lr,reconstruct_coef_1,reconstruct_coef_3,generator_coef,discriminator_coef]
            plot_batch(images, 'train', epoch, coefs)

            image1_plot, image3_plot, image5_plot, image6_plot, label1_plot, label3_plot, is_calligraphy_plot = loader(imageNameTest,idxesTest_1,idxesTest_2,styleTestNum,charTestNum,desired_height=image_size,desired_width=image_size,value_range=(0.0, 1.0),augment=0,force_grayscale=force_grayscale)
            feed_dict_not_training = {image1:image1_plot,image3:image3_plot,image5:image5_plot,image6:image6_plot,label1:label1_plot,label3:label3_plot,is_calligraphy:is_calligraphy_plot,is_training:False}
            _image1_forward_reconstruct,_image3_forward_reconstruct,_image1_style_reconstruct,_image3_style_reconstruct = sess.run([image1_forward_reconstruct,image3_forward_reconstruct,image1_style_reconstruct,image3_style_reconstruct],feed_dict=feed_dict_not_training)
            images = [image1_plot,image3_plot,image5_plot,image6_plot,_image1_forward_reconstruct,_image3_forward_reconstruct,_image1_style_reconstruct,_image3_style_reconstruct]
            coefs = [loss_type,lr,reconstruct_coef_1,reconstruct_coef_3,generator_coef,discriminator_coef]
            plot_batch(images, 'test', epoch, coefs)

            if (epoch+1) % save_frequency == 0:
                coefs = [loss_type,lr,reconstruct_coef_1,reconstruct_coef_2,reconstruct_coef_3,generator_coef,discriminator_coef]
                suffix = ''
                for coef in coefs:
                    suffix += str(coef)+'-'
                saver.save(sess,os.path.join(os.path.join('ckpt',suffix[:-1]),'model'))

        if epoch+1 % save_frequency != 0:
            coefs = [loss_type,lr,reconstruct_coef_1,reconstruct_coef_2,reconstruct_coef_3,generator_coef,discriminator_coef]
            suffix = ''
            for coef in coefs:
                suffix += str(coef)+'-'
            saver.save(sess,os.path.join(os.path.join('ckpt',suffix[:-1]),'model'))


if __name__ == '__main__':
    main()
    