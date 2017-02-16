# DNN From Scratch
# Sean Wade
from __future__ import division, print_function, absolute_import

import numpy as np
from pynet.layers import *
from pynet.utils.data import load_data # uses Keras to load data
from pynet.optim import sgd

# Hyperparameters
REG = 0
BATCH_SIZE = 32
INPUT_DIM = 28 * 28
WEIGHT_SCALE=.1
HIDDEN_DIM = 100
NUM_CLASSES = 10
EPOCHS = 10

# Load data
x_train, y_train, x_test, y_test, y_train_num, y_test_num = load_data('mnist')
print('\nLoaded Data:')
print('x_train: ', x_train.shape) # (60000, 28, 28, 1)
print('y_train: ', y_train.shape) # (10000, 28, 28, 1)
print('x_test: ', x_test.shape)# (60000, 10)
print('y_test: ', y_test.shape) # (10000, 10)

# Initialize weights to train
params = {}
params['W1'] = np.random.normal(scale=WEIGHT_SCALE, size=(INPUT_DIM, HIDDEN_DIM))
params['W2'] = np.random.normal(scale=WEIGHT_SCALE, size=(HIDDEN_DIM, NUM_CLASSES))
params['b1'] = np.zeros(HIDDEN_DIM)
params['b2'] = np.zeros(NUM_CLASSES)

print('\nInitialized Paramaters:')
for name, tensor in params.iteritems():
    print('%s: ' % name, tensor.shape)

# Trianing
num_training = x_train.shape[0]
iters_per_epoch = int(max(num_training / BATCH_SIZE, 1))
num_iterations = int(EPOCHS * iters_per_epoch)

# For logging results
loss_hist = []
acc_hist = []

print('\nStarting Training:')
for epoch in range(EPOCHS):
    # Shuffle the data
    shuffle_idx = np.random.permutation(range(x_train.shape[0]))
    x_train = x_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    y_train_num = y_train_num[shuffle_idx]

    for i in range(iters_per_epoch):
        # Make minibatch
        batch_start = epoch*i
        batch_end = epoch*i + BATCH_SIZE
        x_batch = x_train[batch_start:batch_end]
        y_batch = y_train[batch_start:batch_end]
        y_batch_num = y_train_num[batch_start:batch_end]

        # Forward Pass
        h1, linear_cache_1 = linear_forward(x_batch, params['W1'], params['b1'])
        a1, relu_cache_1= relu_forward(h1)
        h2, linear_cache_2 = linear_forward(a1, params['W2'], params['b2'])
        a2, relu_cache_2 = relu_forward(h2)

        # Loss
        probs, loss, dx = softmax_loss(a2, y_batch_num)
        # Plus regularization
        loss += .5 * REG * np.sum(params['W1']**2) + .5 * REG * np.sum(params['W2']**2)
        loss_hist.append(loss)

        # Backwards Pass
        grads = {}
        da = relu_backward(dx, relu_cache_2)
        dx_2, grads['W2'], grads['b2'] = linear_backward(da, linear_cache_2)
        da_2 = relu_backward(dx_2, relu_cache_1)
        dx, grads['W1'], grads['b1'] = linear_backward(da_2, linear_cache_1)

        # Regularization (optional)
        grads['W2'] += REG * params['W2']
        grads['W1'] += REG * params['W1']

        # Parameter update
        for p, w in params.iteritems():
            dw = grads[p]
            next_w, next_config = sgd(w, dw)
            # Update weights
            params[p] = next_w
        
        # Calculate the accuracy
        if i % 1000 == 0:
            probs = np.exp(a2 - np.max(a2, axis=1, keepdims=True))
            probs /= np.sum(probs, axis=1, keepdims=True)
            y_pred = np.argmax(probs, axis=1)
            train_acc = np.mean(y_pred == y_batch_num)
            acc_hist.append(train_acc)
            print("[{}] loss: {}, Acc: {}".format(epoch, loss, train_acc))

print('\n---Completed Training---')

