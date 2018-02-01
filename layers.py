# TF layers tutorial at:
# https://www.tensorflow.org/tutorials/layers
#
# This is an alternate MNIST CNN.
#
# Date: 29-01-2018
import sys
import os
import shutil
import argparse
from string import Template
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

NFIL1 = 16
NFIL2 = 32
NDENSE = 128
BATCH_SZ = 100
RESULTS = Template('Iter: $global_step   Loss: $loss   Accuracy: $accuracy')

def cnn_model_fn(features, labels, mode, params):
    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])
    biasInit = tf.constant_initializer(0.1, tf.float32)
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=NFIL1,
        kernel_size=5,
        padding='same',
        activation=tf.nn.relu,
        bias_initializer=biasInit)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, 
                                    pool_size=2,
                                    strides=2)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=NFIL2,
        kernel_size=5,
        padding='same',
        activation=tf.nn.relu,
        bias_initializer=biasInit)
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=2,
                                    strides=2)
    pool2flat = tf.reshape(pool2, shape=[-1, 7 * 7 * NFIL2])
    dense = tf.layers.dense(inputs=pool2flat,
                            units=NDENSE,
                            activation=tf.nn.relu,
                            bias_initializer=biasInit)
    dropout = tf.layers.dropout(inputs=dense,
                                rate=params['dropout'],
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
    tf.summary.scalar('accuracy', acc[1])
    tf.summary.histogram('logits', logits)
    tf.summary.histogram('pool1', pool1)
    tf.summary.histogram('pool2', pool2)

    # train op for TRAIN mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learn_rate'])
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

def main(args):
    mnist = input_data.read_data_sets('../data/MNIST_data/', validation_size=0)
    train_data = mnist.train.images
    train_labels = mnist.train.labels.astype(np.int32)
    eval_data = mnist.test.images
    eval_labels = mnist.test.labels.astype(np.int32)

    params = {'dropout': args.dropout,
              'learn_rate': args.learn_rate}
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                              model_dir=args.model_dir,
                                              params=params)
    
    tr_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {'x': train_data},
        y = train_labels,
        batch_size=BATCH_SZ,
        num_epochs=None,
        shuffle=True
    )

    tr_size = len(mnist.train.labels)
    batch_per_epoch = int(tr_size / BATCH_SZ)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {'x': eval_data},
        y = eval_labels,
        num_epochs=1,
        shuffle=False
    )

    for i in range(args.epochs):
        mnist_classifier.train(input_fn=tr_input_fn,
                               steps=batch_per_epoch)
        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print(RESULTS.substitute(eval_results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run train/eval on MNIST')
    parser.add_argument('--epochs', type=int, default=1,
        help='number of training epochs')
    parser.add_argument('--dropout', type=float, default=0.5, 
        help='dropout rate (not retention); 0.0 is no dropout; default 0.5')
    parser.add_argument('--learn_rate', type=float, default=0.001,
        help='Learning rate; default 0.001')
    parser.add_argument('--name', default='',
        help='model directory is <base_dir>/<name>; default '' for <base_dir>')
    parser.add_argument('--base_dir', default='/tmp/layers_example',
        help='base of estimator model_dir; default /tmp/layers_example/')
    parser.add_argument('--overwrite', action='store_true',
        help='Overwrite any model at <base_dir>/<name>, ' 
            + 'otherwise continue training it, if present')
    parser.add_argument('--quiet', action='store_true', 
        help='emit minimal logging information')
    args, _ = parser.parse_known_args(sys.argv)
    args.model_dir = os.path.join(args.base_dir, args.name, '')
    if not args.quiet:
        tf.logging.set_verbosity(tf.logging.INFO)
    if os.path.exists(args.model_dir):
        if args.overwrite:
            shutil.rmtree(args.model_dir)
    main(args)
