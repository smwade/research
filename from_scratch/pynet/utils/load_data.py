from keras.datasets import mnist, cifar10
from keras.utils import np_utils

class Load_data():

    def __init__(self, name='mnist'):
        """ Load data, either mnist or cifar10 """
        if name == 'mnist':
            (x_train, y_train_num), (x_test, y_test_num) = mnist.load_data()
            x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
            x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
            x_train /= 255
            x_test /= 255
            y_train = np_utils.to_categorical(y_train_num, 10)
            y_test = np_utils.to_categorical(y_test_num, 10)

        if name == 'cifar10':
            (x_train, y_train_num), (x_test, y_test_num) = cifar10.load_data()
            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            y_train = np_utils.to_categorical(y_train_num, 10)
            y_test = np_utils.to_categorical(y_test_num, 10)

        return x_train, y_train, x_test, y_test

if __name__ is '__main__':
    load_data('mnist')
    load_data('cifar10')
