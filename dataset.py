# MNIST Classification v1.0 - Dataset | 2020 | EPOCH Foundation
# This work is licensed under the GNU GENERAL PUBLIC LICENSE

# Project ID: bjR42WW1

# Official configs for EPOCH_mnist.

#----------------------------------------------------------------------------

import tensorflow as tf

#----------------------------------------------------------------------------

def load_mnist():
    # Download MNIST Dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Each image is mapped to a single label.
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    #----------------------------------------------------------------------------

    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255.0
    x_test /= 255.0

    # Reshaping the array to 4-dims so that it can work with the Keras API
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    return (x_train, y_train), (x_test, y_test), class_names
