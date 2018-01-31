# TF layers tutorial at:
# https://www.tensorflow.org/tutorials/layers
#
# This is an alternate MNIST CNN.
#
# Date: 29-01-2018
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.INFO)

# Constants
NFIL1 = 16
NFIL2 = 32
NDENSE = 128
DROP = 0.4
LR = 0.001
BATCH_SZ = 100
NUM_EPOCHS = 3
MODEL_DIR = '/tmp/layers_example'

# I'm pretty sure this needs a bias initializer
# It might also need a weights initializer
def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])
    biasInit = tf.constant_initializer(0.1, tf.float32)
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=NFIL1,
        kernel_size=[5, 5],             # try changing to just 5
        padding='same',
        activation=tf.nn.relu,
        bias_initializer=biasInit)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, 
                                    pool_size=[2, 2],   # try just 2
                                    strides=2)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=NFIL2,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu,
        bias_initializer=biasInit)
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[2, 2],
                                    strides=2)
    pool2flat = tf.reshape(pool2, shape=[-1, 7 * 7 * NFIL2])
    dense = tf.layers.dense(inputs=pool2flat,
                            units=NDENSE,
                            activation=tf.nn.relu,
                            bias_initializer=biasInit)
    dropout = tf.layers.dropout(inputs=dense,
                                rate=DROP,
                                training=(mode==tf.estimator.ModeKeys.TRAIN))
    logits = tf.layers.dense(inputs=dropout, units=10)

    # Predictions used in PREDICT and EVAL modes
    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Loss used in EVAL and TRAIN modes
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    acc = tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])

    # train op for TRAIN mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=LR)
        train_op = optimizer.minimize(loss=loss, 
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, 
                                          loss=loss, 
                                          train_op=train_op)

    # EVAL mode
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {'accuracy': acc}
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)

def main():
    mnist = input_data.read_data_sets('../data/MNIST_data/', validation_size=0)
    train_data = mnist.train.images
    train_labels = mnist.train.labels.astype(np.int32)
    eval_data = mnist.test.images
    eval_labels = mnist.test.labels.astype(np.int32)

    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                              model_dir=MODEL_DIR)
    
    tr_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {'x': train_data},
        y = train_labels,
        batch_size=BATCH_SZ,
        num_epochs=None,
        shuffle=True
    )

    #tensors_to_log = {'probabilities': 'softmax_tensor'}
    #logging_hook = tf.train.SummarySaverHook(tensors=tensors_to_log, every_n_iter=100)

    tr_size = len(mnist.train.labels)
    batch_per_epoch = int(tr_size / BATCH_SZ)
    num_batches = NUM_EPOCHS * batch_per_epoch
    mnist_classifier.train(input_fn=tr_input_fn,
                           steps=num_batches)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {'x': eval_data},
        y = eval_labels,
        num_epochs=1,
        shuffle=False
    )
    raw_preds = mnist_classifier.predict(input_fn=eval_input_fn)
    preds = [cl['classes'] for cl in raw_preds]
    print(preds[:500:20])
    print(eval_labels[:500:20].tolist())
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == '__main__':
    # TODO: kill the model tree
    main()