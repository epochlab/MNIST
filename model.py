# MNIST Classification v1.0 - Model | 2020 | EPOCH Foundation
# This work is licensed under the GNU GENERAL PUBLIC LICENSE

# Project ID: bjR42WW1

# Official configs for EPOCH_mnist.

#----------------------------------------------------------------------------

import tensorflow as tf

#----------------------------------------------------------------------------

def build_model():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(32, kernel_size=5, activation='relu', input_shape=(28, 28, 1))) #Convolutional layer to identify features from input
    model.add(tf.keras.layers.MaxPooling2D((2,2))) #Choose the best features via pooling
    model.add(tf.keras.layers.BatchNormalization()) #Normalize the batch to help reduce overfitting during training
    model.add(tf.keras.layers.Dropout(0.2)) #Randomly turn neurons on and off to improve convergence

    model.add(tf.keras.layers.Conv2D(64, kernel_size=5, activation='relu')) #Repeat at 64
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    # A fully connected layer to get all relevant data with rectified linear unit activation, previously was Dense(128) after Flatten().
    model.add(tf.keras.layers.Conv2D(128, kernel_size=4, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())

    # Flatten since too many dimensions, we only want a classification output
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.4))

    model.add(tf.keras.layers.Dense(10)) # Value of 10 because of number class 0 to 9.

    # Compile the model and define loss function, optimizer and metrics.
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    return model
