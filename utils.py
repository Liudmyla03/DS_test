"""
Helper utilities: load MNIST dataset and preprocess
"""

from tensorflow.keras.datasets import mnist
import numpy as np


def load_mnist_flat(normalize: bool = True, as_channels: bool = False):
    """
    Load MNIST and return numpy arrays
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    if normalize:
        x_train /= 255.0
        x_test /= 255.0
    if as_channels:
        x_train = x_train[..., None]
        x_test = x_test[..., None]
    return x_train, y_train, x_test, y_test
