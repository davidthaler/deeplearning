# Equivalent to layers.py, but with keras instead of bare tensorflow
#
# Date: 2-Feb-2018
import argparse
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras import backend

# Build MNIST data; y comes in as a vector of labels in 0...9
(xtr, ytr), (xte, yte) = mnist.load_data()
xtr = xtr.reshape(-1, 28, 28, 1)
ytr = to_categorical(ytr)
xte = xte.reshape(-1, 28, 28, 1)
yte = to_categorical(yte)

# This with statement seems to suppress a TF session-closing issue.
with backend.get_session():
    # Build model
    model = Sequential()
    model.add(Conv2D(input_shape=(28, 28, 1),
                    filters=16,
                    kernel_size=5,
                    activation='relu'))
    # Defaults are shape=(2,2) strides=(2,2), which is what we want
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=32, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    # Default weights initializer is glorot_uniform, which is ok
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # fit/evaluate/print
    model.fit(xtr, ytr, epochs=1, batch_size=100)
    results = model.evaluate(xte, yte)
    print('Loss: %.4f   Acc: %.4f' % tuple(results))
