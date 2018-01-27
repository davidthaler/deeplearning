# MNIST example code from:
# https://www.tensorflow.org/get_started/mnist/beginners
#
# Date: 23-01-2018

import sys
import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def run():
    # load data
    mnist = input_data.read_data_sets('../data/MNIST_data/', one_hot=True)

    # model set-up
    x = tf.placeholder(tf.float32, [None, 784])     # "None" means unspecified dimension
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b                         # logits
    t = tf.placeholder(tf.float32, [None, 10])      # holds ground truth

    # optimization set-up
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=y))
    #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)

    # Create session and initialize variables
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    # Train    
    for _ in range(1000):
        xs, ts = mnist.train.next_batch(100)
        #sess.run(train_step, feed_dict={x:xs, t:ts})
        train_step.run(feed_dict={x:xs, t:ts})

    # Test
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))        # booleans
    acc = tf.reduce_mean(tf.cast(correct, tf.float32))

    # Output
    print(sess.run(acc, feed_dict={x:mnist.test.images, t:mnist.test.labels}))

if __name__ == '__main__':
    run()