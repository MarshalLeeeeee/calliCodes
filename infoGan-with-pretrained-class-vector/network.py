import tensorflow as tf
import tensorflow.contrib.layers as layers
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

def encoder(image,kernel,stride,class_dim,is_training,name):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        conv1 = conv2d(image,32,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'conv1')
        conv2 = conv2d(conv1,64,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'conv2')
        conv3 = conv2d(conv2,128,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'conv3')
        conv4 = conv2d(conv3,128,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'conv4')
        sp = conv4.get_shape()
        flatten = tf.reshape(conv4, [-1,sp[1]*sp[2]*sp[3]])
        fc1 = fc(flatten,1024,is_training,layers.batch_norm,leaky_rectify,'fc1')
        class_vector = fc(fc1,class_dim,is_training,layers.batch_norm,leaky_rectify,'class_vector')
        return class_vector, sp[1], sp[2], sp[3]

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
        return pred, fc2

def mutual_fc(hidden, contiuous_dim, is_training, name):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        fc1 = fc(hidden,128,is_training,layers.batch_norm,leaky_rectify,'fc1')
        fc2 = fc(fc1,contiuous_dim,is_training,None,tf.identity,'fc2')
        return fc2

def infogan(zc_vector,image,kernel,stride,class_dim,style_dim,contiuous_dim,is_training,reconstruct_coef,continuous_coef,generator_coef,discriminator_coef,name):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        image_reconstruct = decoder(zc_vector,4,4,128,kernel,stride,is_training,'decoder')

        reconstruct_loss = reconstruct_coef * tf.reduce_mean(tf.reduce_sum(tf.abs(image-image_reconstruct),[1,2,3]))

        image_pred_true, image_hidden_true = discriminator(image,kernel,stride,is_training,'discriminator')
        image_pred_forward_fake, image_hidden_forward_fake = discriminator(image_reconstruct,kernel,stride,is_training,'discriminator')

        discriminator_true = tf.reduce_mean(tf.log(image_pred_true + 1e-6))
        discriminator_forward_fake = tf.reduce_mean(tf.log(1.0 - image_pred_forward_fake + 1e-6))
        discriminator_forward_true = tf.reduce_mean(tf.log(image_pred_forward_fake + 1e-6))

        true_continuous = tf.slice(zc_vector,[0,class_dim+style_dim], [-1,contiuous_dim])
        mutual_continuous = mutual_fc(image_hidden_forward_fake,contiuous_dim,is_training,'mutual_fc')
        std_contig = tf.ones_like(mutual_continuous)
        epsilon = (true_continuous - mutual_continuous) / std_contig
        continuous_loss = -continuous_coef * tf.reduce_mean(tf.reduce_sum(- 0.5 * np.log(2 * np.pi) - tf.log(std_contig) - 0.5 * tf.square(epsilon),reduction_indices=1))

        generator_loss = -generator_coef * discriminator_forward_true
        discriminator_loss = -discriminator_coef * (discriminator_true + discriminator_forward_fake)

        return reconstruct_loss, continuous_loss, generator_loss, discriminator_loss, image_reconstruct