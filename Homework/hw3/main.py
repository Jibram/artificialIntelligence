# -*- coding: utf-8 -*-
"""LeNet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YssvoRPqUHwFPrzcBogzTZ2DiiAEnW4W
"""

#Name = Jibram

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def run_cnn():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Learning rate was a hyperparameter, 0.01 doesnt really work. Loss stays same. Other values sort of standard.
    learning_rate = 0.0001
    epochs = 10
    batch_size = 100
    
    # input x - for 28 x 28 pixels = 784 - this is the flattened image data that is drawn from mnist.train.nextbatch()
    x = tf.placeholder(tf.float32, [None, 28 * 28])
    
    # 4D Tensor. -1 = dynamic shape. 28x28 image. 1 for grayscale (3 for rgb)
    x_shaped = tf.reshape(x, [-1, 28, 28, 1])
    
    # This is where we will hold our 10 output values
    y = tf.placeholder(tf.float32, [None, 10])

    # CONV and POOL twice
    
    # This creates 4 feature maps. CONV with 4 5x5 kernel. POOL with 2x2 MAX.
    # CONV creates 24x24x4 from 28x28x1. MAX POOL creates 12x12x4 due to stride 2 2x2 kernel
    layer1 = create_new_conv_layer(x_shaped, 1, 4, [5, 5], [2, 2], name='layer1')
    
    # This creates 12 feature maps. CONV with 12 5x5 kernel. POOl with 2x2 MAX. 
    # CONV creates 8x8x12 from 12x12x4. MAX POOL creates 4x4x12 due to stride 2 2x2 kernel
    layer2 = create_new_conv_layer(layer1, 4, 12, [5, 5], [2, 2], name='layer2')
    
    # Output should now be 4x4x12

    # We must vectorize our output to single dimension by "flattening"
    flattened = tf.reshape(layer2, [-1, 4 * 4 * 12])

    # setup some weights and bias values for this flattened layer, then activate with ReLU
    wd1 = tf.Variable(tf.truncated_normal([4 * 4 * 12, 1000], stddev=0.03), name='wd1')
    bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
    dense_layer1 = tf.matmul(flattened, wd1) + bd1
    dense_layer1 = tf.nn.relu(dense_layer1)

    # This is the weights and biases for the 10 weights and the softmax classifier
    wd2 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.03), name='wd2')
    bd2 = tf.Variable(tf.truncated_normal([10], stddev=0.01), name='bd2')
    dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
    y_ = tf.nn.softmax(dense_layer2)

   
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))

    # add an optimiser
    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # define an accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # setup the initialisation operator
    init_op = tf.global_variables_initializer()

    # add a summary to store the accuracy
    tf.summary.scalar('accuracy', accuracy)

    # training loop
    with tf.Session() as sess:
        # initialise the variables
        sess.run(init_op)
        total_batch = int(len(mnist.train.labels) / batch_size)
        total_avg_cost = 0
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
                _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch
            total_avg_cost += avg_cost
            test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            
            #Required print per epoch
            print("Train set:", "Epoch:", (epoch+1), ", Average loss:", "{:.3f}".format(avg_cost), ", lr:{:.4f}".format(learning_rate))

        total_avg_cost /= epochs
        accvalue = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("\nTest set: Average loss: ", "{:.4f}".format(total_avg_cost), ", Accuracy: ", int(len(mnist.train.labels)*accvalue) , "/",len(mnist.train.labels)," (", "{:.2f}".format(accvalue* 100), "%)")


def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # varaible to help compute nn.conv2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

    # init weights and bias
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # run the convolution on the input data
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='VALID')

    # add the bias
    out_layer += bias

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # =====MAX POOLING=====
    # ksize is the argument which defines the size of the max pooling window (i.e. the area over which the maximum is
    # calculated).  It must be 4D to match the convolution - in this case, for each image we want to use a 2 x 2 area
    # applied to each channel
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    # strides defines how the max pooling area moves through the image - a stride of 2 in the x direction will lead to
    # max pooling areas starting at x=0, x=2, x=4 etc. through your image.  If the stride is 1, we will get max pooling
    # overlapping previous max pooling areas (and no reduction in the number of parameters).  In this case, we want
    # to do strides of 2 in the x and y directions.
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='VALID')

    return out_layer

if __name__ == "__main__":
    run_cnn()
