# Equivalent to layers.py, but with keras instead of bare tensorflow
#
# NB: there is an intermittent TF session-closing bug/issue in keras/TF.
# The internet knows about it, but does not have a conclusive fix.
#
# Date: 2-Feb-2018
import sys
import argparse
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras import backend
from keras.initializers import Constant


def get_mnist():
    # Build MNIST data
    (xtr, ytr), (xte, yte) = mnist.load_data()
    # x comes in as uint8 0...255; float from 0...1.0 works better
    xtr = xtr.reshape(-1, 28, 28, 1) / 256.0
    # y comes in as a vector of labels in 0...9
    ytr = to_categorical(ytr)
    xte = xte.reshape(-1, 28, 28, 1) / 256.0
    yte = to_categorical(yte)
    return xtr, ytr, xte, yte

def main(args):
    xtr, ytr, xte, yte = get_mnist()
    # This with statement may help suppress a TF session-closing issue.
    with backend.get_session():
        # Build model
        model = Sequential()
        model.add(Conv2D(input_shape=(28, 28, 1),
                        filters=args.filters1,
                        kernel_size=5,
                        activation='relu',
                        bias_initializer=Constant(args.init)))
        # Defaults are shape=(2,2) strides=(2,2), which is what we want
        model.add(MaxPooling2D())
        model.add(Conv2D(filters=args.filters2,
                        kernel_size=5,
                        activation='relu',
                        bias_initializer=Constant(args.init)))
        model.add(MaxPooling2D())
        model.add(Flatten())
        # Default weights initializer is glorot_uniform, which is ok
        model.add(Dense(args.dense,
                        activation='relu',
                        bias_initializer=Constant(args.init)))
        model.add(Dropout(rate=args.dropout))
        model.add(Dense(10, activation='softmax'))
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        # fit/evaluate/print
        model.fit(xtr, ytr, epochs=args.epochs, batch_size=100)
        results = model.evaluate(xte, yte)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run keras CNN on MNIST data')
    parser.add_argument('--epochs', type=int, default=1,
        help='number of training epochs; default 1')
    parser.add_argument('--dropout', type=float, default=0.5, 
        help='dropout rate (not retention); 0.0 is no dropout; default 0.5')
    parser.add_argument('--filters1', type=int, default=16,
        help='Number of filters in first convolutional layer; default 16')
    parser.add_argument('--filters2', type=int, default=32,
        help='Number of filters in second convolutional layer; default 32')
    parser.add_argument('--dense', type=int, default=128,
        help='Number of units in dense, fully-connected layer; default 128')
    parser.add_argument('--init', type=float, default=0.0,
        help='Bias initializer value for relu layers; default 0.0.')
    args, _ = parser.parse_known_args(sys.argv)
    results = main(args)
    print('Loss: %.4f   Acc: %.4f' % tuple(results))
