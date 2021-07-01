#!/usr/bin/env python3

import os, datetime

import numpy as np
import tensorflow as tf

from dataset import load_mnist
from model import build_model

print("Version:", tf.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

(x_train, y_train), (x_test, y_test), class_names = load_mnist()
print("MNIST dataset loaded")

print('Training Dataset Shape:', x_train.shape)
print('Number of training images:', x_train.shape[0])
print("Number of training labels:", (len(y_train)))
print('Number of test images:', x_test.shape[0])
print("Number of test labels:", (len(y_test)))

model = build_model()
model.summary()

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

checkpoint_path = "saved_models/" + timestamp + "/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
              filepath=checkpoint_path,
              save_weights_only=True,
              monitor='val_loss',
              save_best_only=True,
              verbose=0)

annealer = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x, verbose=0)

csv_log = tf.keras.callbacks.CSVLogger("saved_models/" + timestamp + ".log", separator=",", append=True)

datagen = tf.keras.preprocessing.image.ImageDataGenerator( rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1)

EPOCHS = 20
BATCH_SIZE = 64

model_log = model.fit(datagen.flow(x=x_train, y=y_train, batch_size=BATCH_SIZE),
                      epochs=EPOCHS,
                      steps_per_epoch = x_train.shape[0]//BATCH_SIZE,
                      validation_data=(x_test, y_test),
                      verbose=1,
                      callbacks=[annealer, cp_callback, csv_log])

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print('Test accuracy: {:5.2f}%'.format(100*test_acc))
print('Test loss:', test_loss)

latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)

loss,acc = model.evaluate(x_test,  y_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
