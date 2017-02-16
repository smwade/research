from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
import time
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

# Load Data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/MNIST_data', one_hot=True)

for depth in range(5,10):
    for width in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
        model_name = 'depth_{}-width_{}'.format(depth, width)
        print('\n\n--- STARTING ' + model_name + ' ---\n\n')
        model = Sequential()
        model.add(Dense(output_dim=width, input_dim=784, init='he_normal'))
        model.add(Activation("relu"))
        for _ in range(depth-1):
            model.add(Dense(output_dim=width, init='he_normal'))
            model.add(Activation("relu"))
        model.add(Dense(output_dim=10, init='he_normal'))
        model.add(Activation("softmax"))

        # Chose Loss and Compile
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        # Callbacks
        csv_logger = keras.callbacks.CSVLogger('./logs/' + model_name + '.log')
        early_stop = keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, verbose=False, mode='auto')
        history = keras.callbacks.History()
        tensorboard = keras.callbacks.TensorBoard(log_dir='./tf-logs/'+model_name, histogram_freq=0, write_graph=True, write_images=True)
        callback_list = [csv_logger, early_stop, history, tensorboard]

        # Train Model
        model.fit(mnist.train.images, mnist.train.labels, nb_epoch=100, batch_size=32, validation_data=(mnist.test.images, mnist.test.labels), callbacks=callback_list)
