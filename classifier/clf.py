import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import argparse
from loader import load_calligraphy
import random
import os

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help="gpu to use")
    parser.add_argument('--gpu_fraction', type=float, default=0.4, help="fraction of gpu memory to use")
    parser.add_argument('--categorical_cardinality', type=int, default=100, help="number of the characters to be loaded")
    parser.add_argument('--data_path', type=str, default='../../demo/', help="path to save images")
    parser.add_argument('--styles', type=str, default='cklxz', help="calligraphy style (sub folders)")
    parser.add_argument('--image_size', type=int, default=64, help="the size of images trained")
    parser.add_argument('--force_grayscale', type=bool, default=True, help="transform images into single channel or not")
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('--lr', type=float, default=5e-2, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--epochs', type=int, default=200, help="epochs")
    parser.add_argument('--kernel', type=int, default=4, help="kernel size")
    parser.add_argument('--stride', type=int, default=2, help="stride")
    return parser.parse_args()

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

def conv2d(inputs,num_outputs,kernel_size,stride,is_training,normalizer_fn,activation_fn,scope,name):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        out = layers.convolution2d(inputs,
                    num_outputs=num_outputs,
                    kernel_size=kernel_size,
                    stride=stride,
                    normalizer_params={"is_training": is_training},
                    normalizer_fn=normalizer_fn,
                    activation_fn=activation_fn,
                    scope=scope)
        return out

def fc(inputs,num_outputs,is_training,normalizer_fn,activation_fn,scope,name):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        out = layers.fully_connected(inputs,
                num_outputs=num_outputs,
                activation_fn=activation_fn,
                normalizer_fn=normalizer_fn,
                normalizer_params={"is_training": is_training, "updates_collections": None},
                scope=scope)
        return out

def network(image, kernel, stride, categorical_cardinality, is_training,name='clf'):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        conv1 = conv2d(image,32,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'layer_1','conv_1')
        conv2 = conv2d(conv1,64,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'layer_2','conv_2')
        conv3 = conv2d(conv2,128,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'layer_3','conv_3')
        conv4 = conv2d(conv3,128,kernel,stride,is_training,conv_batch_norm,leaky_rectify,'layer_4','conv_4')
        sp = conv4.get_shape()
        flatten = tf.reshape(conv4, [-1,sp[1]*sp[2]*sp[3]])
        fc1 = fc(flatten,1024,is_training,layers.batch_norm,leaky_rectify,'layer_5','fc_1')
        out = fc(fc1,categorical_cardinality,is_training,layers.batch_norm,leaky_rectify,'layer_6','fc_2')
        return out

def main():
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

    X_train, Y_train, X_test, Y_test = load_calligraphy(data_path,categorical_cardinality,styles,image_size,image_size,(0,1),force_grayscale)
    length_train, length_test = X_train.shape[0], X_test.shape[0]
    print(length_train, length_test)

    image = tf.placeholder(tf.float32,[None, image_size, image_size, channel_size],name="input_images")
    label = tf.placeholder(tf.float32,[None, categorical_cardinality],name="input_label")
    is_training = tf.placeholder(tf.bool,[],name="is_training")


    pred = network(image,kernel,stride,categorical_cardinality,is_training)
    entropy_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=pred))
    solver = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.5)
    train = solver.minimize(entropy_loss, var_list=scope_variables('clf'))

    config = tf.ConfigProto() 
    config.gpu_options.per_process_gpu_memory_fraction = parser.gpu_fraction
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        idxes_train = np.arange(length_train, dtype=np.int32)
        idxes_test = np.arange(length_test, dtype=np.int32)
        accuracies = []
        for epoch in range(epochs):
            np.random.shuffle(idxes_train)
            np.random.shuffle(idxes_test)
            losses = []
            corrects = 0
            for idx in range(0,length_train,batch_size):
                image_batch_train = X_train[idxes_train[idx:idx + batch_size]]
                label_batch_train = Y_train[idxes_train[idx:idx + batch_size]]
                _,loss_train = sess.run([train,entropy_loss],feed_dict={image:image_batch_train,label:label_batch_train,is_training:True})

            for idx in range(0,length_test,batch_size):
                image_batch_test = X_test[idxes_test[idx:idx + batch_size]]
                label_batch_test = Y_test[idxes_test[idx:idx + batch_size]]
                loss_test, pred_test = sess.run([entropy_loss, pred],feed_dict={image:image_batch_test,label:label_batch_test,is_training:False})
                losses.append(loss_test)
                pred_category = np.argmax(pred_test, axis=1)
                true_category = np.argmax(label_batch_test, axis=1)
                corrects += np.sum(pred_category == true_category)
                
            accuracies.append(corrects / length_test)
            print('epoch %d finished, loss on test set is %f, accuracy is %f' % (epoch, np.mean(losses), accuracies[-1]))

        accuracies = np.array(accuracies)
        with open('log','a') as f:
            f.write('===============================================\n')
            f.write('categorical cardinality: %d \n' % categorical_cardinality)
            f.write('lr: %f \n' % lr)
            f.write('batch size: %d \n' % batch_size)
            f.write('epochs: %d \n' % epochs)
            f.write('kernel size : %d \n' % kernel)
            f.write('stride: %d \n' % stride)
            f.write('mean accuracy for the last 10 epochs: %f \n' % np.mean(accuracies[-10:]))
            f.write('\n')


if __name__ == '__main__':
	main()