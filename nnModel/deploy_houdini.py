#!/usr/bin/env python3

import sys

import numpy as np
import tensorflow as tf

from PIL import Image

from dataset import load_mnist
from model import build_model

checkpoint_dir = ''

raw_input = int(sys.argv[1])
print('Raw Input: ' + str(raw_input))

(x_train, y_train), (x_test, y_test), class_names = load_mnist()
print("MNIST dataset loaded")

model = build_model()

latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)

loss,acc = model.evaluate(x_test,  y_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
predictions = probability_model.predict(x_test)

layers = [3, 7, 8]

for layer in layers:
    out = model.layers[layer].output
    print(out)

    feature_map = tf.keras.Model(inputs = model.inputs, outputs = out)
    feature_map.summary()

    for x in range(raw_input):
        fmap=feature_map.predict(x_test[x].reshape(1,28,28,1))
        print(fmap.shape)

        if layer == 3:
            features = 32
        if layer == 7:
            features = 64
        if layer == 8:
            features = 128

        OUTDIR = 'featureMaps/Conv2D_' + str(features) + '/'

        for i in range(features):
            data = fmap[0,:,:,i]
            rescaled = (255.0 * data).astype(np.uint8)

            im = Image.fromarray(rescaled)
            im.save(OUTDIR + str(x) + '/featureMap_' + str(features) + '_i' + str(i) + '.png')
