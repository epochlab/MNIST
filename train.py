# MNIST Classification v1.0 - Train | 2020 | EPOCH Foundation
# This work is licensed under the GNU GENERAL PUBLIC LICENSE

# Project ID: bjR42WW1

# Official configs for EPOCH_mnist.

#----------------------------------------------------------------------------

import os
import datetime

import numpy as np
import tensorflow as tf

from dataset import load_mnist
from model import build_model

#----------------------------------------------------------------------------

print("Version:", tf.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

#----------------------------------------------------------------------------

(x_train, y_train), (x_test, y_test), class_names = load_mnist()
print("MNIST dataset loaded")

#----------------------------------------------------------------------------

print('Training Dataset Shape:', x_train.shape)
print('Number of training images:', x_train.shape[0])
print("Number of training labels:", (len(y_train)))
print('Number of test images:', x_test.shape[0])
print("Number of test labels:", (len(y_test)))

#----------------------------------------------------------------------------

model = build_model()
model.summary()

#----------------------------------------------------------------------------

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Create a callback that saves epoch weights during training.
checkpoint_path = "saved_models/" + timestamp + "/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
              filepath=checkpoint_path,
              save_weights_only=True,
              monitor='val_loss',
              save_best_only=True,
              verbose=0)

# Decrease the learning rate each epoch.
annealer = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x, verbose=0)

# Write logs to .csv
csv_log = tf.keras.callbacks.CSVLogger("saved_models/" + timestamp + ".log", separator=",", append=True)

# Create a larger dataset by using data augmentation.
datagen = tf.keras.preprocessing.image.ImageDataGenerator( rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1)

#----------------------------------------------------------------------------

# Train the model over a number of epochs with callback and batch size.
epochs = 20
batch_size = 64

model_log = model.fit(datagen.flow(x=x_train, y=y_train, batch_size=batch_size),
                      epochs=epochs,
                      steps_per_epoch = x_train.shape[0]//batch_size,
                      validation_data=(x_test, y_test),
                      verbose=1,
                      callbacks=[annealer, cp_callback, csv_log])

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print('Test accuracy: {:5.2f}%'.format(100*test_acc))
print('Test loss:', test_loss)

#----------------------------------------------------------------------------

# Loads the weights
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)

# Re-evaluate the pre-trained model
loss,acc = model.evaluate(x_test,  y_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
