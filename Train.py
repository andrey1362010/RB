import os
import pickle

import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from RleUtils import rle2mask
from generateTraining import DataLoader

dataLoader = DataLoader()
#batch, result = dataLoader.get()
#print(batch.shape)
#fig = plt.figure(figsize=(2, 1))
#fig.add_subplot(2, 1, 1)
#plt.imshow(batch[2, 6, :, :, 0])
#fig.add_subplot(2, 1, 2)
#plt.imshow(np.transpose(result[2], (1, 0)))
#plt.show()


input = tf.placeholder(tf.float32, (None, 21, 21, 384, 1), name='input')
result_placeholder = tf.placeholder(tf.float32, (None, 384, 8), name='input')

# Layer 1
layer = tf.layers.conv3d(input, 8, kernel_size=3, strides=2, padding="same")
layer = tf.layers.batch_normalization(layer, training=True)
layer = tf.nn.leaky_relu(layer)
# Layer 2
layer = tf.layers.conv3d(layer, 16, kernel_size=3, strides=(1, 1, 2), padding="same")
layer = tf.layers.batch_normalization(layer, training=True)
layer = tf.nn.leaky_relu(layer)
# Layer 3
layer = tf.layers.conv3d(layer, 32, kernel_size=3, strides=(1, 1, 2), padding="same")
layer = tf.layers.batch_normalization(layer, training=True)
layer = tf.nn.leaky_relu(layer)
# Layer 4
layer = tf.layers.conv3d(layer, 64, kernel_size=3, strides=(1, 1, 2), padding="same")
layer = tf.layers.batch_normalization(layer, training=True)
layer = tf.nn.leaky_relu(layer)
# Final Layer
layer = tf.reduce_mean(layer, [1, 2], keep_dims=True)

layer = tf.layers.conv3d_transpose(layer, 64, kernel_size=(1, 1, 3), strides=(1, 1, 2), padding="same")
layer = tf.layers.batch_normalization(layer, training=True)
layer = tf.nn.leaky_relu(layer)

layer = tf.layers.conv3d_transpose(layer, 32, kernel_size=(1, 1, 3), strides=(1, 1, 2), padding="same")
layer = tf.layers.batch_normalization(layer, training=True)
layer = tf.nn.leaky_relu(layer)

layer = tf.layers.conv3d_transpose(layer, 16, kernel_size=(1, 1, 3), strides=(1, 1, 2), padding="same")
layer = tf.layers.batch_normalization(layer, training=True)
layer = tf.nn.leaky_relu(layer)

layer = tf.layers.conv3d_transpose(layer, 8, kernel_size=(1, 1, 3), strides=(1, 1, 2), padding="same")
logist = tf.squeeze(layer, axis=[1, 2])



loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=result_placeholder, logits=logist))
opt = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(100000):
        print("Epoch:", epoch)
        for i in range(1000):
            batch, res = dataLoader.get()
            l, _ = sess.run([loss, opt], feed_dict={input: batch, result_placeholder: res})
            print("Loss:", l)
