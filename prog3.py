# Third example program at:
# https://www.tensorflow.org/get_started/get_started
#
# Date: 22 January 2018
import numpy as np
import tensorflow as tf

def model_fn(features, labels, mode):
    '''
    Create LR model with the low-level API, for use as a tf.estimator

    Args:
        features: iterable of tf.feature_column, must have key 'x'
        labels: 
        mode: one of 'train', 'predict', 'eval', roughly speaking

    Returns:
        some object
    '''
    w = tf.get_variable('w', [1], dtype=tf.float64)
    b = tf.get_variable('b', [1], dtype=tf.float64)
    y = w * features['x'] + b
    loss = tf.reduce_sum(tf.square(y - labels))
    global_step = tf.train.get_global_step()
    opt = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(opt.minimize(loss), tf.assign_add(global_step, 1))

    # This is where we make the tf.estimator object
    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=y,
                                      loss=loss,
                                      train_op=train)

# our tf.estimator.Estimator class model
model = tf.estimator.Estimator(model_fn=model_fn)

# data (from prog2.py)
xtr = np.array([1., 2., 3., 4.])
ytr = np.array([0., -1., -2., -3.])
xte = np.array([2., 5., 8., 1.])
yte = np.array([-1.01, -4.1, -7., 0.])

# Copied from prog2.py
input_fn = tf.estimator.inputs.numpy_input_fn({'x':xtr}, ytr, batch_size=4, 
                                                num_epochs=None, shuffle=True)
tr_input_fn = tf.estimator.inputs.numpy_input_fn({'x': xtr}, ytr, batch_size=4, 
                                                num_epochs=1000, shuffle=False)
val_input_fn = tf.estimator.inputs.numpy_input_fn({'x': xte}, yte, batch_size=4, 
                                                num_epochs=1000, shuffle=False)

# train (same as prog2.py)
model.train(input_fn=input_fn, steps=1000)

# Copied from prog2.py
tr_results = model.evaluate(input_fn=tr_input_fn)
val_results = model.evaluate(input_fn=val_input_fn)
print('Train results: %r' % tr_results)
print('Validation results: %r' % val_results)