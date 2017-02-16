# CNN From Scratch
# Sean Wade
from __future__ import division, print_function, absolute_import

import numpy as np
import pickle
from pynet.layers import *
from pynet.utils.data import load_data # uses Keras to load data
from pynet.optim import adam

# Hyperparameters
REG = 0
BATCH_SIZE = 32
WEIGHT_SCALE=.1
NUM_CLASSES = 10
PAD = 0
STRIDE = 1
EPOCHS = 10

# Load data
x_train, y_train, x_test, y_test, y_train_num, y_test_num = load_data('mnist')
x_train = np.swapaxes(x_train,1,3)
x_test = np.swapaxes(x_test,1,3)
print('\nLoaded Data:')
print('x_train: ', x_train.shape) # (60000, 28, 28, 1)
print('y_train: ', y_train.shape) # (10000, 28, 28, 1)
print('x_test: ', x_test.shape)# (60000, 10)
print('y_test: ', y_test.shape) # (10000, 10)

# Initialize weights to train
params = {}
params['W1'] = np.random.normal(scale=WEIGHT_SCALE, size=(32, 1, 3, 3))
params['W2'] = np.random.normal(scale=WEIGHT_SCALE, size=(32, 32, 3, 3))
params['W3'] = np.random.normal(scale=WEIGHT_SCALE, size=(24*24*32, NUM_CLASSES))
params['b1'] = np.zeros(32)
params['b2'] = np.zeros(32)
params['b3'] = np.zeros(NUM_CLASSES)

print('\nInitialized Paramaters:')
for name, tensor in params.iteritems():
    print('%s: ' % name, tensor.shape)

# For logging results
loss_hist = []
acc_hist = []
test_acc_hist = []

print('\nStarting Training:')
for epoch in range(EPOCHS):
    # Shuffle the data
    shuffle_idx = np.random.permutation(range(x_train.shape[0]))
    x_train = x_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    y_train_num = y_train_num[shuffle_idx]

    for i in range(int(x_train.shape[0] / BATCH_SIZE)):
        # Make minibatch
        batch_start = epoch*i
        batch_end = epoch*i + BATCH_SIZE
        x_batch = x_train[batch_start:batch_end]
        y_batch = y_train[batch_start:batch_end]
        y_batch_num = y_train_num[batch_start:batch_end]

        # Forward Pass
        # 1st Conv Layer
        h1, conv_cache_1 = conv_forward(x_batch, params['W1'], params['b1'], pad=PAD, stride=STRIDE)
        a1, relu_cache_1= relu_forward(h1)
        # 2nd Conv Layer
        h2, conv_cache_2 = conv_forward(a1, params['W2'], params['b2'], pad=PAD, stride=STRIDE)
        a2, relu_cache_2 = relu_forward(h2)
        # Fully Connected Layer
        h3, linear_cache_3 = linear_forward(a2, params['W3'], params['b3'])
        a3, relu_cache_3 = relu_forward(h3)

        # Loss
        probs, loss, dx = softmax_loss(a3, y_batch_num)
        # Plus regularization
        loss += .5 * REG * np.sum(params['W1']**2) + .5 * REG * np.sum(params['W2']**2) \
                + .5 * REG * np.sum(params['W3']**2)
        loss_hist.append(loss)

        # Backwards Pass
        grads = {}
        da_3 = relu_backward(dx, relu_cache_3)
        dx_2, grads['W3'], grads['b3'] = linear_backward(da_3, linear_cache_3)
        da_2 = relu_backward(dx_2, relu_cache_2)
        dx_3, grads['W2'], grads['b2'] = conv_backward(da_2, conv_cache_2)
        da_1 = relu_backward(dx_3, relu_cache_1)
        dx, grads['W1'], grads['b1'] = conv_backward(da_1, conv_cache_1)

        # Regularization (optional)
        grads['W3'] += REG * params['W3']
        grads['W2'] += REG * params['W2']
        grads['W1'] += REG * params['W1']

        # Parameter update
        for p, w in params.iteritems():
            dw = grads[p]
            next_w, next_config = adam(w, dw)
            # Update weights
            params[p] = next_w
        
        # Calculate the accuracy
        if i % 10 == 0:
            probs = np.exp(a3 - np.max(a3, axis=1, keepdims=True))
            probs /= np.sum(probs, axis=1, keepdims=True)
            y_pred = np.argmax(probs, axis=1)
            train_acc = np.mean(y_pred == y_batch_num)
            acc_hist.append(train_acc)
            print("[{}] loss: {}, Acc: {}".format(i, loss, train_acc))

    # Calc Test Accuracy
    h1, conv_cache_1 = conv_forward(x_test, params['W1'], params['b1'], pad=PAD, stride=STRIDE)
    a1, relu_cache_1= relu_forward(h1)
    h2, conv_cache_2 = conv_forward(a1, params['W2'], params['b2'], pad=PAD, stride=STRIDE)
    a2, relu_cache_2 = relu_forward(h2)
    h3, linear_cache_3 = linear_forward(a2, params['W3'], params['b3'])
    a3, relu_cache_3 = relu_forward(h3)
    probs = np.exp(a3 - np.max(a3, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    y_pred = np.argmax(probs, axis=1)
    test_acc = np.mean(y_pred == y_test)
    test_acc_hist.append(test_acc)

    print("[{}] Epoch Complete: ".format(epoch))
    print("Test Acc: {}".format(test_acc))

# Save results and model
pickle.dump(params, open('model_weights.p', 'wb'))
pickle.dump(acc_hist, open('cnn_acc_list.p','wb'))
pickle.dump(loss_hist, open('cnn_loss_list.p','wb'))
pickle.dump(test_acc, open('cnn_test_acc.p','wb'))

print('\n---Completed Training---')
