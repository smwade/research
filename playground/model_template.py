#!/usr/bin/env python

# A general framework for a keras model
# Sean Wade

from __future__ import absolute_import, division, print_function
import numpy as np
import pickle
import datetime
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import keras
from keras.datasets import cifar10, mnist
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from seansUtils.research import StatsCallback 

# Hyper Parameters
# --------------------------
TRAIN = True
BATCH_SIZE = 32
EPOCHS = 300
DATA = 'mnist'
MODEL_NAME = 'better_cnn'
FILEPATH = './saved/'

if not os.path.exists(FILEPATH):
    os.makedirs(FILEPATH)

# Load/Prep the Data
# --------------------------
print('Loading the data %s...' % DATA)
if DATA == 'mnist':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
else:
    (x_train, y_train_num), (x_test, y_test_num) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = np_utils.to_categorical(y_train_num, 10)
    y_test = np_utils.to_categorical(y_test_num, 10)
print('Data Loaded.')

# Model
# --------------------------
model = Sequential()

model.add(Convolution2D(32, 3, 3, input_shape=(28, 28, 1)))
model.add(Activation('relu'))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))

# Loss and Optimizer
# --------------------------
adam = Adam() # <--- Adjust optimizer here
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Callbacks
# --------------------------
# pick the callbacks you want
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
checkpoint = keras.callbacks.ModelCheckpoint(FILEPATH + MODEL_NAME, monitor='val_loss', verbose=False, save_best_only=True)
csv_logger = keras.callbacks.CSVLogger(FILEPATH + MODEL_NAME + '.log')
stats = StatsCallback(MODEL_NAME)
lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, \
        mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
callback_list = [stats, lr, checkpoint, early_stopping]

# Train the Model
# --------------------------
print('... Starting Training ...')
model.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, \
        validation_data=(x_test, y_test), verbose=True, callbacks=callback_list)

# Save Model and Stats
# --------------------------
print('Saving results')
stats = stats.stats_dict
pickle.dump(stats, open(FILEPATH + MODEL_NAME + '_stats', 'wb'))
model.save(FILEPATH + MODEL_NAME)
print('---COMPLETE---')
