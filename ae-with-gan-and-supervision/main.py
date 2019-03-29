import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import argparse
import random
import os
from PIL import Image

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
    parser.add_argument('--style_dim', type=int, default=75, help="dimension of style vector")
    parser.add_argument('--reconstruct_coef', type=float, default=1.0, help="reconstruct coef")
    parser.add_argument('--generator_coef', type=float, default=1.0, help="generator coef")
    parser.add_argument('--discriminator_coef', type=float, default=1.0, help="discriminator coef")
    return parser.parse_args()

def locate(data_path, styles=None, max_label=100):
    imageName, imageDict = [], {}
    if styles is None: styles = ['std-comp']
    for i in range(len(styles)):
        path = os.path.join(data_path,styles[i])
        for basepath, directories, fnames in os.walk(path):
            for fname in fnames:
                flabel = int(fname.split('/')[-1].split('.')[0].split('-')[0])
                if flabel < max_label:
                    imageName.append(os.path.join(basepath,fname))
                    if flabel not in imageDict: imageDict[flabel] = []
                    imageDict[flabel].append(os.path.join(basepath,fname))
    return np.array(imageName), imageDict

def choice(image1,imageDict):
    image2 = []
    for fname in image1:
        flabel = int(fname.split('/')[-1].split('.')[0].split('-')[0])
        image2.append(np.random.choice(imageDict[flabel]))
    return image2

def find_truth(image1,imageTrue):
    image2 = []
    for fname in image1:
        flabel = int(fname.split('/')[-1].split('.')[0].split('-')[0])
        image2.append(imageTrue[flabel][0])
    return image2

def loader(imageName,desired_height,desired_width,value_range,force_grayscale=True):
    image_batch = None
    for fname in imageName:
        image = Image.open(fname)
        width, height = image.size
        if width != desired_width or height != desired_height:
            image = image.resize((desired_width, desired_height), Image.BILINEAR)
        if force_grayscale: 
            image = image.convert("L")
        img = np.array(image)
        if len(img.shape) == 2: 
            img = img[:, :, None]
        if image_batch is None: 
            image_batch = np.array([img], dtype=np.float32)
        else: 
            image_batch = np.concatenate((image_batch,np.array([img], dtype=np.float32)),axis=0)
    image_batch = (value_range[0] + (image_batch / 255.0) * (value_range[1] - value_range[0]))
    return image_batch

def plot(image1_plot, image1_reconstruct, image2_plot, epoch):
    num, w, h, c = image1_plot.shape[0], image1_plot.shape[1], image1_plot.shape[2], image1_plot.shape[3]
    img = Image.new('L',(w*3,h*num))
    for i in range(num):
        img.paste(Image.fromarray(np.squeeze((image1_plot[i]*255).astype(np.uint8))),(64*0,64*i))
        img.paste(Image.fromarray(np.squeeze((image1_reconstruct[i]*255).astype(np.uint8))),(64*1,64*i))
        img.paste(Image.fromarray(np.squeeze((image2_plot[i]*255).astype(np.uint8))),(64*2,64*i))
    img.save(os.path.join('savedImages',str(epoch)+'.png'))


def get_mean(x):
    return np.mean(np.array(x, dtype=np.float32))

def variables_in_current_scope():
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)

def scope_variables(name):
    with tf.variable_scope(name):
        return variables_in_current_scope()

def leaky_rectify(x, leakiness=0.01):
    assert leakiness <= 1
    ret = tf.maximum(x, leakiness * x)
    return ret

def conv_batch_norm(inputs,
                    name="batch_norm",
                    is_training=True,
                    trainable=True,
                    epsilon=1e-5):
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    shp = inputs.get_shape()[-1].value

    with tf.variable_scope(name) as scope:
        gamma = tf.get_variable("gamma", [shp], initializer=tf.random_normal_initializer(1., 0.02), trainable=trainable)
        beta = tf.get_variable("beta", [shp], initializer=tf.constant_initializer(0.), trainable=trainable)

        mean, variance = tf.nn.moments(inputs, [0, 1, 2])
        mean.set_shape((shp,))
        variance.set_shape((shp,))
        ema_apply_op = ema.apply([mean, variance])

        def update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.nn.batch_norm_with_global_normalization(
                    inputs, mean, variance, beta, gamma, epsilon,
                    scale_after_normalization=True
                )
        def do_not_update():
            return tf.nn.batch_norm_with_global_normalization(
                inputs, ema.average(mean), ema.average(variance), beta,
                gamma, epsilon,
                scale_after_normalization=True
            )

        normalized_x = tf.cond(
            is_training,
            update,
            do_not_update
        )
        return normalized_x

def conv2d(inputs,num_outputs,kernel_size,stride,is_training,normalizer_fn,activation_fn,name):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        out = layers.convolution2d(inputs,
                    num_outputs=num_outputs,
                    kernel_size=kernel_size,
                    stride=stride,
                    normalizer_params={"is_training": is_training},
                    normalizer_fn=normalizer_fn,
                    activation_fn=activation_fn)
        return out

def conv2d_transpose(inputs,num_outputs,kernel_size,stride,is_training,normalizer_fn,activation_fn,name):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        out = layers.convolution2d_transpose(inputs,
                    num_outputs=num_outputs,
                    kernel_size=kernel_size,
                    stride=stride,
                    normalizer_params={"is_training": is_training},
                    normalizer_fn=normalizer_fn,
                    activation_fn=activation_fn)
        return out

def fc(inputs,num_outputs,is_training,normalizer_fn,activation_fn,name):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        out = layers.fully_connected(inputs,
                num_outputs=num_outputs,
                activation_fn=activation_fn,
                normalizer_fn=normalizer_fn,
                normalizer_params={"is_training": is_training, "updates_collections": None})
        return out

def encoder(image,kernel,stride,class_dim,style_dim,is_training,name):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        conv1 = conv2d(image,32,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'conv1')
        conv2 = conv2d(conv1,64,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'conv2')
        conv3 = conv2d(conv2,128,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'conv3')
        conv4 = conv2d(conv3,128,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'conv4')
        sp = conv4.get_shape()
        flatten = tf.reshape(conv4, [-1,sp[1]*sp[2]*sp[3]])
        fc1 = fc(flatten,1024,is_training,layers.batch_norm,leaky_rectify,'fc1')
        class_vector = fc(fc1,class_dim,is_training,layers.batch_norm,leaky_rectify,'class_vector')
        style_vector = fc(fc1,style_dim,is_training,layers.batch_norm,leaky_rectify,'style_miu')
        return class_vector, style_vector, sp[1], sp[2], sp[3]

def decoder(vector,w,h,c,kernel,stride,is_training,name):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        fc1 = fc(vector,1024,is_training,layers.batch_norm,leaky_rectify,'fc1')
        fc2 = fc(fc1,w*h*c,is_training,layers.batch_norm,leaky_rectify,'fc2')
        expand = tf.reshape(fc2, [-1,w,h,c])
        conv1 = conv2d_transpose(expand,128,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'conv1')
        conv2 = conv2d_transpose(conv1,64,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'conv2')
        conv3 = conv2d_transpose(conv2,32,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'conv3')
        conv4 = conv2d_transpose(conv3,1,kernel,stride,is_training,conv_batch_norm,tf.nn.sigmoid,'conv4')
        return conv4

def discriminator(image,kernel,stride,is_training,name):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        conv1 = conv2d(image,32,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'conv1')
        conv2 = conv2d(conv1,64,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'conv2')
        conv3 = conv2d(conv2,128,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'conv3')
        conv4 = conv2d(conv3,128,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'conv4')
        sp = conv4.get_shape()
        flatten = tf.reshape(conv4, [-1,sp[1]*sp[2]*sp[3]])
        fc1 = fc(flatten,1024,is_training,layers.batch_norm,leaky_rectify,'fc1')
        fc2 = fc(fc1,128,is_training,layers.batch_norm,leaky_rectify,'fc2')
        pred = fc(fc2,1,is_training,layers.batch_norm,tf.nn.sigmoid,'pred')
        return pred

def cycle_consistent_vae_with_gan(image1,image2,kernel,stride,class_dim,style_dim,is_training,reconstruct_coef,generator_coef,discriminator_coef,name):
    # image1, image2, imgae3 are all batched image data
    # specifically, every item in image1 has the same class with its corresponding one in image2
    # while image3 is independent to image1 (randomly picked)
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        class_vector_1, style_vector_1, w, h, c = encoder(image1,kernel,stride,class_dim,style_dim,is_training,'encoder')
        w,h,c = int(w), int(h), int(c)
        
        vector_1 = tf.concat([class_vector_1, style_vector_1], 1)

        # forward
        image1_forward_reconstruct = decoder(vector_1,w,h,c,kernel,stride,is_training,'decoder')

        reconstruct_loss_1 = tf.reduce_mean(tf.reduce_sum(tf.abs(image2-image1_forward_reconstruct),[1,2,3]))
        reconstruct_loss = reconstruct_coef * reconstruct_loss_1
        forward_loss = reconstruct_loss

        image1_pred_true = discriminator(image2,kernel,stride,is_training,'discriminator')
        image1_pred_forward_fake = discriminator(image1_forward_reconstruct,kernel,stride,is_training,'discriminator')

        discriminator_true_1 = tf.reduce_mean(tf.log(image1_pred_true + 1e-6))
        discriminator_forward_fake_1 = tf.reduce_mean(tf.log(1.0 - image1_pred_forward_fake + 1e-6))
        discriminator_forward_true_1 = tf.reduce_mean(tf.log(image1_pred_forward_fake + 1e-6))

        generator_loss = -generator_coef * discriminator_forward_true_1
        discriminator_loss = -discriminator_coef * (discriminator_true_1 + discriminator_forward_fake_1)

        return forward_loss, reconstruct_loss, generator_loss, discriminator_loss, image1_forward_reconstruct

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
    style_dim = parser.style_dim
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

    forward_loss, reconstruct_loss, generator_loss, discriminator_loss, image1_forward_reconstruct = cycle_consistent_vae_with_gan(
                                                                                                      image1,image2,kernel,stride,class_dim,style_dim,is_training,
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


if __name__ == '__main__':
    main()
    '''
    X = (np.random.sample((10,10))*255).astype(np.uint8)
    img2 = Image.open('x.png')
    img2_arr = np.array(img2)
    print(img2_arr.shape, img2_arr.dtype)
    img = Image.fromarray(img2_arr)
    img.save('y.png')
    print(X.shape, X.dtype)
    print(X)
    img = Image.fromarray(X)
    img.save('try.png')
    '''
    