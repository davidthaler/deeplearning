# Second example program at:
# https://www.tensorflow.org/get_started/get_started
#
# Date: 22 January 2018
import numpy as np
import tensorflow as tf

# set-up
f = [tf.feature_column.numeric_column('x')]
model = tf.estimator.LinearRegressor(feature_columns=f)

# data
xtr = np.array([1., 2., 3., 4.])
ytr = np.array([0., -1., -2., -3.])
xte = np.array([2., 5., 8., 1.])
yte = np.array([-1.01, -4.1, -7., 0.])

# Looks like we use 'input functions' to deal the data
input_fn = tf.estimator.inputs.numpy_input_fn({'x':xtr}, ytr, batch_size=4, 
                                                num_epochs=None, shuffle=True)
tr_input_fn = tf.estimator.inputs.numpy_input_fn({'x': xtr}, ytr, batch_size=4, 
                                                num_epochs=1000, shuffle=False)
val_input_fn = tf.estimator.inputs.numpy_input_fn({'x': xte}, yte, batch_size=4, 
                                                num_epochs=1000, shuffle=False)
model.train(input_fn=input_fn, steps=1000)
tr_results = model.evaluate(input_fn=tr_input_fn)
val_results = model.evaluate(input_fn=val_input_fn)
print('Train results: %r' % tr_results)
print('Validation results: %r' % val_results)
