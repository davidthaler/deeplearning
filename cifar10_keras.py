# First crack at cifar10 with keras
# At end of first day, this gets just under 70% test in 10 epochs at defaults.
#
# Date: 03-Feb-2018
import os
import sys
import argparse
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout
from keras.layers import Flatten, Activation, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

def get_cifar():
    (xtr, ytr), (xte, yte) = cifar10.load_data()
    # y is a vector of labels in 0...9; change to one-hot
    ytr = to_categorical(ytr)
    yte = to_categorical(yte)
    # x is uint8 0...255; change to float in [0.0, 1.0]
    xtr = (xtr / 255.0).astype('float32')
    xte = (xte / 255.0).astype('float32')
    return xtr, ytr, xte, yte

def main(args):
    xtr, ytr, xte, yte = get_cifar()
    model = Sequential()
    model.add(Conv2D(input_shape=(32, 32, 3),
                     filters=args.filters1,
                     kernel_size=5))
    if args.batchnorm:
        model.add(BatchNormalization(scale=False))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=args.filters2,
                     kernel_size=3))
    if args.batchnorm:
        model.add(BatchNormalization(scale=False))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=args.filters3,
                     kernel_size=3))
    if args.batchnorm:
        model.add(BatchNormalization(scale=False))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(args.dense))
    if args.batchnorm:
        model.add(BatchNormalization(scale=False))
    model.add(Activation('relu'))
    model.add(Dropout(rate=args.dropout))
    model.add(Dense(10))
    if args.batchnorm:
        model.add(BatchNormalization(scale=False))
    model.add(Activation('softmax'))
    model.compile(optimizer=Adam(lr=args.learn_rate, decay=args.lr_decay),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    callbacks = None
    if args.name != '':
        log_dir = os.path.join(args.base_dir, args.name, '')
        histogram = 1 if (args.histogram and args.validation > 0.0) else 0
        callbacks = [TensorBoard(log_dir=log_dir, histogram_freq=histogram)]
    model.fit(xtr, ytr,
              batch_size=args.batch_sz,
              epochs=args.epochs,
              validation_split=args.validation,
              callbacks=callbacks)
    results = model.evaluate(xte, yte)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do train/eval on cifar10')
    parser.add_argument('--batch_sz', type=int, default=100,
        help='training batch size; default 100')
    parser.add_argument('--epochs', type=int, default=1,
        help='number of training epochs; default 1')
    parser.add_argument('--dropout', type=float, default=0.5, 
        help='dropout rate (not retention); 0.0 is no dropout; default 0.5')
    parser.add_argument('--filters1', type=int, default=32,
        help='Number of filters in first convolutional layer; default 32')
    parser.add_argument('--filters2', type=int, default=64,
        help='Number of filters in second convolutional layer; default 64')
    parser.add_argument('--filters3', type=int, default=128,
        help='Number of filters in second convolutional layer; default 128')
    parser.add_argument('--dense', type=int, default=128,
        help='Number of units in dense, fully-connected layer; default 128')
    parser.add_argument('--name', default='',
        help='model directory is <base_dir>/<name>; default '' for <base_dir>')
    parser.add_argument('--base_dir', default='/tmp/cifar10/',
        help='base of estimator model_dir; default /tmp/cifar10/')
    parser.add_argument('--validation', type=float, default=0.0, 
        help='fraction of training data to use for validation; default 0.0')
    parser.add_argument('--histogram', action='store_true',
        help='log histograms for TensorBoard; validation must be set to > 0')
    parser.add_argument('--batchnorm', action='store_true', 
        help='Apply batchnorm on all layers')
    parser.add_argument('--learn_rate', type=float, default=0.001,
        help='Learning rate; default 0.001; ' +
        'NB: w/Adam optimizer this parameter is less important.')
    parser.add_argument('--lr_decay', type=float, default=0.0,
        help='Learning rate decay; default 0.0; ' +
        'NB: w/Adam optimizer this parameter is less important.')
    args, _ = parser.parse_known_args()
    results = main(args)
    print('Loss: %.4f   Acc: %.4f' % tuple(results))
