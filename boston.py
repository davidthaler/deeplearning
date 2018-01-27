# Tensorflow example from:
#  https://www.tensorflow.org/get_started/input_fn
#
# NB: It appears that this model is *wildly* unstable as provided.
# So don't panic if the results of one run are crap.
#
# Also, this saves its work at model_dir, and then starts over.
# So you might want to rm -rf <MODEL_DIR>
#
# Date 23-01-2018
import itertools
import pandas as pd
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = ['crim', 'zn', 'indus', 'nox', 'rm', 'age',
           'dis', 'tax', 'ptratio', 'medv']
FEATURES = ['crim', 'zn', 'indus', 'nox', 'rm',
            'age', 'dis', 'tax', 'ptratio']
LABEL = 'medv'
MODEL_DIR = '/tmp/boston_model'

tr = pd.read_csv('../data/boston_train.csv', skiprows=1, names=COLUMNS)
te = pd.read_csv('../data/boston_test.csv', skiprows=1, names=COLUMNS)
feature_columns = [tf.feature_column.numeric_column(f) for f in FEATURES]

def get_input_fn(dataset, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
        x=dataset[FEATURES],
        y=dataset[LABEL],
        num_epochs=num_epochs, shuffle=shuffle)

# NB: the loss numbers emitted are total loss (400 examples)
model = tf.estimator.DNNRegressor(feature_columns=feature_columns,
                                  hidden_units=[10, 10],
                                  model_dir=MODEL_DIR)
model.train(input_fn=get_input_fn(tr), steps=5000)
ev = model.evaluate(input_fn=get_input_fn(te, num_epochs=1, shuffle=False))
print("Loss: {0:f}".format(ev['average_loss']))

pr = pd.read_csv('../data/boston_predict.csv', skiprows=1, names=COLUMNS)
pred = model.predict(input_fn=get_input_fn(pr, num_epochs=1, shuffle=False))
predictions = [p['predictions'][0] for  p in itertools.islice(pred, 6)]
print("Predictions: {}".format(str(predictions)))
