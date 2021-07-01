# MNIST Classification v1.0 - Evaluate | 2020 | EPOCH Foundation
# This work is licensed under the GNU GENERAL PUBLIC LICENSE

# Project ID: bjR42WW1

# Official configs for EPOCH_mnist.

#----------------------------------------------------------------------------

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from dataset import load_mnist
from model import build_model

#----------------------------------------------------------------------------

# Generate graph to review full set of class predictions.
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

#----------------------------------------------------------------------------

checkpoint_dir = 'saved_models/20201010-003224/'

#----------------------------------------------------------------------------

(x_train, y_train), (x_test, y_test), class_names = load_mnist()
print("MNIST dataset loaded")

#----------------------------------------------------------------------------

# Create new untrained model to valiadate callback
model = build_model()
loss, acc = model.evaluate(x_test,  y_test, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

# Loads the weights
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)

# Re-evaluate the pre-trained model
loss,acc = model.evaluate(x_test,  y_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

#----------------------------------------------------------------------------

# Create inference function and seed to evaluate.
probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])

i = 1234

predictions = probability_model.predict(x_test)
print("\nPrediction Array:", predictions[i])
print("Argmax:", np.argmax(predictions[i]))
print("Label Class:", y_test[i])

#----------------------------------------------------------------------------

# Verify predictions using plot fuctions.
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], y_test, x_test.reshape(x_test.shape[0], 28, 28))
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  y_test)
plt.show()

# Plot serveral images as a test array to ensure confidence.
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], y_test, x_test.reshape(x_test.shape[0], 28, 28))
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], y_test)
plt.tight_layout()
plt.show()
