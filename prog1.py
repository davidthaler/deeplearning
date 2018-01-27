# First example program from:
# https://www.tensorflow.org/get_started/get_started
#
# Date: 22 January 2018
import tensorflow as tf

w = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([-0.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
lm = w * x + b
y = tf.placeholder(tf.float32)
loss = tf.reduce_sum(tf.square(lm - y))
opt = tf.train.GradientDescentOptimizer(0.01)
train = opt.minimize(loss)
xtr = [1,2,3,4]
ytr = [0, -1, -2, -3]

# Begin training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    sess.run(train, {x:xtr, y:ytr})

wf, bf, lf = sess.run([w, b, loss], {x:xtr, y:ytr})
print('w: %s  b: %s  loss: %s' % (wf, bf, lf))
