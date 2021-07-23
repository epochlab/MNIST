#!/usr/bin/env python3

import os, requests, gzip, hashlib
import numpy as np
import tensorflow as tf

def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255.0
    x_test /= 255.0

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    return (x_train, y_train), (x_test, y_test), class_names

def fetch(url):
    filepath = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(filepath):
        with open(filepath, 'rb') as f:
            data = f.read()
    else:
        with open(filepath, 'wb') as f:
            data = requests.get(url).content
            f.write(data)
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()

x_train = fetch('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
y_train = fetch('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
x_test = fetch('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
y_test = fetch('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')
