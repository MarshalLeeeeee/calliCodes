import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.ops import variable_scope
import numpy as np

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

def cross_entrophy(label,pred,tiny=1e-6):
    return -(label*tf.log(pred+tiny)+(1-label)*tf.log(1-pred+tiny))

def conv_batch_norm(inputs,
                    name="batch_norm",
                    is_training=True,
                    trainable=True,
                    epsilon=1e-5):
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    shp = inputs.get_shape()[-1].value

    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
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

def encoder(image,kernel,stride,class_dim,style_dim,is_training,name='encoder'):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
        print(scope.name, name)
        conv1 = conv2d(image,32,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'conv1')
        conv2 = conv2d(conv1,64,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'conv2')
        conv3 = conv2d(conv2,128,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'conv3')
        conv4 = conv2d(conv3,128,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'conv4')
        sp = conv4.get_shape()
        flatten = tf.reshape(conv4, [-1,sp[1]*sp[2]*sp[3]])
        fc1 = fc(flatten,1024,is_training,layers.batch_norm,leaky_rectify,'fc1')
        #class_vector = fc(fc1,class_dim,is_training,layers.batch_norm,leaky_rectify,'class_vector')
        #style_vector = fc(fc1,style_dim,is_training,layers.batch_norm,leaky_rectify,'style_vector')
        class_vector = fc(fc1,class_dim,is_training,layers.batch_norm,tf.nn.tanh,'class_vector')
        style_vector = fc(fc1,style_dim,is_training,layers.batch_norm,tf.nn.tanh,'style_vector')
    return class_vector, style_vector, sp[1], sp[2], sp[3]

def decoder(vector,w,h,c,kernel,stride,is_training,name='decoder'):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
        print(scope.name, name)
        fc1 = fc(vector,1024,is_training,layers.batch_norm,leaky_rectify,'fc1')
        fc2 = fc(fc1,w*h*c,is_training,layers.batch_norm,leaky_rectify,'fc2')
        expand = tf.reshape(fc2, [-1,w,h,c])
        conv1 = conv2d_transpose(expand,128,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'conv1')
        conv2 = conv2d_transpose(conv1,64,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'conv2')
        conv3 = conv2d_transpose(conv2,32,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'conv3')
        conv4 = conv2d_transpose(conv3,1,kernel,stride,is_training,conv_batch_norm,tf.nn.sigmoid,'conv4')
    return conv4

def discriminator(image,kernel,stride,is_training,classNum=1,name='discriminator'):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
        print(scope.name, name)
        conv1 = conv2d(image,32,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'conv1')
        conv2 = conv2d(conv1,64,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'conv2')
        conv3 = conv2d(conv2,128,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'conv3')
        conv4 = conv2d(conv3,128,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'conv4')
        sp = conv4.get_shape()
        flatten = tf.reshape(conv4, [-1,sp[1]*sp[2]*sp[3]])
        fc1 = fc(flatten,1024,is_training,layers.batch_norm,leaky_rectify,'fc1')
        fc2 = fc(fc1,128,is_training,layers.batch_norm,leaky_rectify,'fc2')
        pred = fc(fc2,classNum,is_training,layers.batch_norm,tf.nn.sigmoid,'pred')
    return pred

def ae_with_gan(image1,image3,image5,image6,label1,label3,is_calligraphy,kernel,stride,class_dim,style_dim,image_size,channel_size,is_training,loss_type,style_num,reconstruct_coef_1,reconstruct_coef_3,generator_coef,discriminator_coef,name='ae-with-gan'):
    # image2 is the ground truth of image1
    # image4 is the ground truth of image3
    # image5 if the ground truth of content of image1 with style of image3
    # image6 if the ground truth of content of image3 with style of image1
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
        class_vector_1, style_vector_1, w, h, c = encoder(image1,kernel,stride,class_dim,style_dim,is_training)
        class_vector_3, style_vector_3, _, _, _ = encoder(image3,kernel,stride,class_dim,style_dim,is_training)
        w,h,c = int(w), int(h), int(c)

        # forward
        image1_forward_reconstruct = decoder(tf.concat([class_vector_1,style_vector_1],1),w,h,c,kernel,stride,is_training)
        image3_forward_reconstruct = decoder(tf.concat([class_vector_3,style_vector_3],1),w,h,c,kernel,stride,is_training)
        image1_forward_reconstruct = binary(image1_forward_reconstruct,image_size,channel_size,0.7)
        image3_forward_reconstruct = binary(image3_forward_reconstruct,image_size,channel_size,0.7)
        if loss_type == 'l1':
            reconstruct_loss_1 = tf.reduce_mean(tf.reduce_sum(tf.abs(image1-image1_forward_reconstruct),[1,2,3])) + tf.reduce_mean(tf.reduce_sum(tf.abs(image3-image3_forward_reconstruct),[1,2,3]))
        elif loss_type == 'ce':
            reconstruct_loss_1 = tf.reduce_mean(cross_entrophy(image1,image1_forward_reconstruct)) + tf.reduce_mean(cross_entrophy(image3,image3_forward_reconstruct))
        else:
            reconstruct_loss_1 = 0
        reconstruct_loss_1 = reconstruct_coef_1 * reconstruct_loss_1

        # swap
        image1_style_reconstruct = decoder(tf.concat([class_vector_1,style_vector_3],1),w,h,c,kernel,stride,is_training)
        image3_style_reconstruct = decoder(tf.concat([class_vector_3,style_vector_1],1),w,h,c,kernel,stride,is_training)
        image1_style_reconstruct = binary(image1_style_reconstruct,image_size,channel_size,0.7)
        image3_style_reconstruct = binary(image3_style_reconstruct,image_size,channel_size,0.7)
        if loss_type == 'l1':
            reconstruct_loss_3 = tf.reduce_mean(tf.reduce_sum(tf.abs(image5-image1_style_reconstruct),[1,2,3])) + tf.reduce_mean(tf.reduce_sum(tf.abs(image6-image3_style_reconstruct),[1,2,3]))
        elif loss_type == 'ce':
            reconstruct_loss_3 = tf.reduce_mean(cross_entrophy(image5,image1_style_reconstruct)) + tf.reduce_mean(cross_entrophy(image6,image3_style_reconstruct))
        else:
            reconstruct_loss_3 = 0
        reconstruct_loss_3 = reconstruct_coef_3 * reconstruct_loss_3

        forward_loss = reconstruct_loss_1 + reconstruct_loss_3

        style_num = style_num + 1
        image1_pred_true = discriminator(image1,kernel,stride,is_training,style_num,'discriminator')
        image3_pred_true = discriminator(image3,kernel,stride,is_training,style_num,'discriminator')
        image5_pred_true = discriminator(image5,kernel,stride,is_training,style_num,'discriminator')
        image6_pred_true = discriminator(image6,kernel,stride,is_training,style_num,'discriminator')
        image1_pred_forward_fake = discriminator(image1_forward_reconstruct,kernel,stride,is_training,style_num,'discriminator')
        image3_pred_forward_fake = discriminator(image3_forward_reconstruct,kernel,stride,is_training,style_num,'discriminator')
        image1_style_forward_fake = discriminator(image1_style_reconstruct,kernel,stride,is_training,style_num,'discriminator')
        image3_style_forward_fake = discriminator(image3_style_reconstruct,kernel,stride,is_training,style_num,'discriminator')

        label1_true = tf.concat([is_calligraphy,label1],1)
        label3_true = tf.concat([is_calligraphy,label3],1)
        label_fake = tf.zeros_like(label1_true)

        discriminator_true_1 = tf.reduce_mean(cross_entrophy(label1_true,image1_pred_true))
        discriminator_true_3 = tf.reduce_mean(cross_entrophy(label3_true,image3_pred_true))
        discriminator_true_5 = tf.reduce_mean(cross_entrophy(label3_true,image5_pred_true))
        discriminator_true_6 = tf.reduce_mean(cross_entrophy(label1_true,image6_pred_true))
        discriminator_forward_fake_1 = tf.reduce_mean(cross_entrophy(label_fake,image1_pred_forward_fake))
        discriminator_forward_true_1 = tf.reduce_mean(cross_entrophy(label1_true,image1_pred_forward_fake))
        discriminator_forward_fake_3 = tf.reduce_mean(cross_entrophy(label_fake,image3_pred_forward_fake))
        discriminator_forward_true_3 = tf.reduce_mean(cross_entrophy(label3_true,image3_pred_forward_fake))
        discriminator_style_fake_1 = tf.reduce_mean(cross_entrophy(label_fake,image1_style_forward_fake))
        discriminator_style_true_1 = tf.reduce_mean(cross_entrophy(label3_true,image1_style_forward_fake))
        discriminator_style_fake_3 = tf.reduce_mean(cross_entrophy(label_fake,image3_style_forward_fake))
        discriminator_style_true_3 = tf.reduce_mean(cross_entrophy(label1_true,image3_style_forward_fake))

        generator_loss_1 = reconstruct_coef_1 * discriminator_forward_true_1 + reconstruct_coef_3 * discriminator_style_true_3
        generator_loss_3 = reconstruct_coef_1 * discriminator_forward_true_3 + reconstruct_coef_3 * discriminator_style_true_1
        generator_loss = generator_coef * (generator_loss_1 + generator_loss_3)

        discriminator_loss_1 = reconstruct_coef_1 * (discriminator_true_1 + discriminator_true_6 + discriminator_forward_fake_1) + reconstruct_coef_3 * discriminator_style_fake_3
        discriminator_loss_3 = reconstruct_coef_1 * (discriminator_true_3 + discriminator_true_5 + discriminator_forward_fake_3) + reconstruct_coef_3 * discriminator_style_fake_1
        discriminator_loss = discriminator_coef * (discriminator_loss_1 + discriminator_loss_3)

    return forward_loss, reconstruct_loss_1, reconstruct_loss_3, generator_loss, discriminator_loss, \
    image1_forward_reconstruct, image3_forward_reconstruct, image1_style_reconstruct, image3_style_reconstruct, \
    class_vector_1, style_vector_1, class_vector_3, style_vector_3,\
    image1_pred_true, image3_pred_true, image5_pred_true, image6_pred_true, image1_pred_forward_fake, image3_pred_forward_fake, image1_style_forward_fake, image3_style_forward_fake

def binary(image,image_size,channel_size,threshold=None):
    image_average = tf.reshape(tf.reduce_mean(image,[1,2,3]),[-1,1,1,1])
    if threshold is not None:
        image_average = tf.cast(tf.ones_like(image_average)*threshold,tf.float32)
    image_average = tf.tile(image_average,[1, image_size, image_size, channel_size])
    image_mask = tf.cast(tf.less(image,image_average),tf.float32)
    return image*image_mask+(1-image_mask)

def transfer(image1,image2,kernel,stride,class_dim,style_dim,is_training,name):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        class_vector_1, style_vector_1, w, h, c = encoder(image1,kernel,stride,class_dim,style_dim,is_training)
        class_vector_2, style_vector_2, w, h, c = encoder(image2,kernel,stride,class_dim,style_dim,is_training)
        w,h,c = int(w), int(h), int(c)

        image1_forward_reconstruct = decoder(tf.concat([class_vector_1,style_vector_2],1),w,h,c,kernel,stride,is_training)
        image2_forward_reconstruct = decoder(tf.concat([class_vector_2,style_vector_1],1),w,h,c,kernel,stride,is_training)

        return image1_forward_reconstruct, image2_forward_reconstruct