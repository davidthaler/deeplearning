# Simplified version of the estimator quick-start example at:
#   https://www.tensorflow.org/get_started/estimator
# 
# Date 22 January 2018
import numpy as np
import tensorflow as tf

TRAIN = '../data/iris_training.csv'
TEST = '../data/iris_test.csv'

# Load Data the tensorflow way...
tr = tf.contrib.learn.datasets.base.load_csv_with_header(filename=TRAIN, 
    target_dtype=np.int, features_dtype = np.float32)
te = tf.contrib.learn.datasets.base.load_csv_with_header(filename=TEST, 
    target_dtype=np.int, features_dtype=np.float32)

# Specify all features real-valued
f = [tf.feature_column.numeric_column('x', shape=[4])]

# Build 3-layer DNN with 10, 20, 10 units
model = tf.estimator.DNNClassifier(feature_columns=f, 
                                   hidden_units=[10, 20, 10],
                                   n_classes=3,
                                   model_dir='/tmp/iris_model')

tr_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': np.array(tr.data)},
                                                 y=np.array(tr.target),
                                                 num_epochs=None,
                                                 shuffle=True)

te_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': np.array(te.data)},
                                                 y=np.array(te.target),
                                                 num_epochs=1,
                                                 shuffle=False)

model.train(input_fn=tr_input_fn, steps=2000)
acc = model.evaluate(input_fn=te_input_fn)['accuracy']
print("\nTest Accuracy: {0:f}\n".format(acc))

# predict on some new data
newdata = np.array([[6.4, 3.2, 4.5, 1.5],
                    [5.8, 3.1, 5.0, 1.7]])
pr_input_fn = tf.estimator.inputs.numpy_input_fn({'x': newdata}, num_epochs=1, shuffle=False)
pred = list(model.predict(input_fn=pr_input_fn))
pred_cls = [p['classes'] for p in pred]
print('Predictions: {}\n'.format(pred_cls))
