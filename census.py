# Extensively modified version of example at:
# https://www.tensorflow.org/tutorials/wide#learn_deeper
# Also partially adapted from boston.py
# 
# Date: 25-01-2018
import pandas as pd
import tensorflow as tf

TRAIN = '../data/census_data/adult.train'
TEST = '../data/census_data/adult.test'
MODEL_DIR = '/tmp/census'

_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]

COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

FEATURES = COLUMNS[: -1]
LABEL = COLUMNS[-1]

def get_input_fn(data, num_epochs=None, shuffle=True):
    # Defaults are for training; use 1 and True for test
    return tf.estimator.inputs.pandas_input_fn(
        x=data[FEATURES], y=(data[LABEL]=='>50K').astype(int), 
        num_epochs=num_epochs, shuffle=shuffle)

def get_feature_cols():
    '''Build the feature columns that define the model.'''
    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education',
        ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
          'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
          '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status',
        ['Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
          'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship',
        ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
          'Other-relative'])

    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass',
        ['Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
          'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

    occupation = tf.feature_column.categorical_column_with_hash_bucket(
        'occupation', hash_bucket_size=1000)

    age = tf.feature_column.numeric_column('age')
    age_buckets = tf.feature_column.bucketized_column(
            age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    base_columns = [education, marital_status, relationship, workclass, occupation, age_buckets]
    return base_columns

def load_data():
    tr = pd.read_csv(TRAIN, header=None, names=COLUMNS)
    te = pd.read_csv(TEST, header=None, names=COLUMNS)
    return tr, te

def run():
    tr, te = load_data()
    model = tf.estimator.LinearClassifier(feature_columns = get_feature_cols(), 
                                          model_dir=MODEL_DIR)
    model.train(input_fn=get_input_fn(tr), steps=10)
    ev = model.evaluate(input_fn=get_input_fn(te, shuffle=False, num_epochs=1))
    return ev

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    ev = run()
    print('Results: %r' % ev)
