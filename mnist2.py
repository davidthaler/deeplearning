# Trying to do a 2-layer (one hidden layer) NN on my own.
# Partially adapted from these 2 sources:
# https://www.tensorflow.org/get_started/mnist/pros
# https://github.com/martin-gorner/tensorflow-mnist-tutorial/blob/master/mnist_2.1_five_layers_relu_lrdecay.py
#
# Date: 26-01-2018
import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../data/MNIST_data/', one_hot=True)

# Number of hidden units
NHIDDEN = 50
NEPOCHS = 20
BATCH_SZ = 100
TR_SZ = len(mnist.train.images)
NBATCHES = int(NEPOCHS * (TR_SZ / BATCH_SZ))

wt_decay = 1e-5
max_lr = 0.003
min_lr = 0.0001
lr_decay_rate = 2000
lr = tf.placeholder(tf.float32)

x = tf.placeholder(tf.float32, [None, 784])
t = tf.placeholder(tf.float32, [None, 10])      # ground-truth
W1 = tf.Variable(tf.truncated_normal([784, NHIDDEN], stddev=0.1))
B1 = tf.Variable(tf.constant(0.1, shape=[NHIDDEN]))
W2 = tf.Variable(tf.truncated_normal([NHIDDEN, 10], stddev=0.1))
B2 = tf.Variable(tf.zeros([10]))
z = tf.nn.relu(tf.matmul(x, W1) + B1)
#z = tf.nn.sigmoid(tf.matmul(x, W1) + B1)       # clearly worse (0.948 vs 0.958 @ 20 HU/20 epochs)
ylogits = tf.matmul(z, W2) + B2
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=ylogits))
#train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)
reg_loss = cross_entropy + wt_decay * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2))
#train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(reg_loss)
train_step = tf.train.AdamOptimizer(lr).minimize(reg_loss)
correct = tf.equal(tf.argmax(ylogits, 1), tf.argmax(t, 1))
acc = tf.reduce_mean(tf.cast(correct, tf.float32))

# Set-up
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1, NBATCHES + 1):
    xs, ts = mnist.train.next_batch(BATCH_SZ)
    learn_rate = min_lr + (max_lr - min_lr) * math.exp(-i/lr_decay_rate)
    sess.run(train_step, feed_dict={x:xs, t:ts, lr:learn_rate})
    if i % 1000 == 0:
        a, c = sess.run([acc, cross_entropy], feed_dict={x:mnist.validation.images, t:mnist.validation.labels})
        print('Batch %d; Val Accuracy: %f; Val Loss: %f; Learn: %f' % (i, a, c, learn_rate))

result = sess.run([acc, cross_entropy], feed_dict={x:mnist.test.images, t:mnist.test.labels})
print('Test Accuracy: %f, Test Loss: %f' % tuple(result))