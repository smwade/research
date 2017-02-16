#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import numpy as np
import argparse
import tensorflow as tf
import os
import pickle
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.utils import np_utils
from seansUtils.research import StatsCallback

SAVEDIR = './dnn-logs/'

def load_data():
    print("Loading data...")
    (x_train, y_train_num), (x_test, y_test_num) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = np_utils.to_categorical(y_train_num, 10)
    y_test = np_utils.to_categorical(y_test_num, 10)
    return x_train, x_test, y_train, y_test

def main(args):
    x_train, x_test, y_train, y_test = load_data()
    print('BUILDING MODEL ---')
    width, depth = args.width, args.depth
    model_name = 'dnn-depth_{}-width_{}'.format(depth, width)
    model = Sequential()
    model.add(Reshape((784,), input_shape=(28, 28, 1)))
    model.add(Dense(output_dim=width, init='he_normal', bias=True))
    model.add(Activation("relu"))
    for _ in range(depth-1):
        model.add(Dense(output_dim=width, init='he_normal', bias=True))
        model.add(Activation("relu"))
    model.add(Dense(output_dim=10, init='he_normal', bias=True))
    model.add(Activation("softmax"))

    # Chose Loss and Compile
    model.compile(loss='categorical_crossentropy', optimizer=args.optimizer, metrics=['accuracy'])

    print('Num Params: %d' % model.count_params())

    # Callbacks
    csv_logger = keras.callbacks.CSVLogger(SAVEDIR + model_name + '.log')
    stats = StatsCallback(model_name, savedir=SAVEDIR+'stats/')
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, verbose=args.verbose, mode='auto')
    history = keras.callbacks.History()
    remote = keras.callbacks.RemoteMonitor(root='http://localhost:9000/')
    tensorboard = keras.callbacks.TensorBoard(log_dir=SAVEDIR+'tf-logs/' + model_name, histogram_freq=0, write_graph=True, write_images=True)

    callback_list = []
    if args.save:
        if not os.path.exists(SAVEDIR):
            os.makedirs(SAVEDIR)
            os.makedirs(SAVEDIR+'models')
            os.makedirs(SAVEDIR+'stats')
        callback_list.append(csv_logger)
        callback_list.append(stats)
    if args.early_stop:
        callback_list.append(early_stop)

    # Train Model
    print('STARTING TRAINING ---')
    model.fit(x_train, y_train, nb_epoch=args.epochs, batch_size=args.batch_size, \
            validation_data=(x_test, y_test), callbacks=callback_list, verbose=args.verbose)

    if args.save:
        print('Saving results...')
        model.save(SAVEDIR+'models/'+model_name+'.h5')

    print('Complete.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make a DNN model with different widths and heights.')
    parser.add_argument('-d', '--depth', help='Depth of the network.', default=2, type=int)
    parser.add_argument('-w', '--width', help='Width of the network hidden layers.', default=100, type=int)
    parser.add_argument('-bs', '--batch_size', help='Batch size for training', default=32, type=int)
    parser.add_argument('-es', '--early_stop', help='Early stopping if training stops.', default='store_true')
    parser.add_argument('-e', '--epochs', help='Number of epochs to train.', default=50, type=int)
    parser.add_argument('-o', '--optimizer', help='Optimizer for training.', default='adam')
    parser.add_argument('-v', '--verbose', help='Verbose output.', action='store_false')
    parser.add_argument('-s', '--save', help='Dont save the results.', action='store_false')

    args = parser.parse_args()
    main(args)



