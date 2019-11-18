import os
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from RleUtils import rle2mask

TRAIN_IMAGES_PATH = "C:/data/rosneft/train"
masks_pd = pd.read_csv("C:/data/rosneft/train_masks.csv")

def read_images_data(path):
    inline_min = 10000
    inline_max = 0
    xline_min = 10000
    xline_max = 0
    for file_name in os.listdir(path):
        type = file_name[:file_name.index("_")]
        index = int(file_name[file_name.index("_") + 1:file_name.index(".")])
        if type == "xline":
            xline_min = min(xline_min, index)
            xline_max = max(xline_max, index)
        if type == "inline":
            inline_min = min(inline_min, index)
            inline_max = max(inline_max, index)
    array = np.zeros((inline_max - inline_min + 1, xline_max - xline_min + 1, 384, 2))
    result_array = np.zeros((inline_max - inline_min + 1, xline_max - xline_min + 1, 384, 8))
    for file_name in os.listdir(path):
        print(file_name)
        img = cv2.imread(os.path.join(TRAIN_IMAGES_PATH, file_name), cv2.IMREAD_GRAYSCALE)
        mask = np.zeros((img.shape[0], img.shape[1], 8))
        for i, rle in enumerate(masks_pd[masks_pd['ImageId'] == file_name]['EncodedPixels']):
            mask[:, :, i+1] = np.array(rle2mask(rle, shape=(img.shape[1], img.shape[0])), dtype=np.float)

        #fig, ax = plt.subplots(1)
        #ax.imshow(mask[:, :, 3:6])
        #plt.show()

        type = file_name[:file_name.index("_")]
        index = int(file_name[file_name.index("_") + 1:file_name.index(".")])
        if type == "xline":
            array[:, index - xline_min, :, 1] = (np.transpose(img, (1, 0)) - 127.5) / 127.5
            result_array[:, index - xline_min, :] = np.transpose(mask, (1, 0, 2))
        if type == "inline":
            array[index - inline_min:, :, :, 0] = (np.transpose(img, (1, 0)) - 127.5) / 127.5
            result_array[index - inline_min:, :, :] = np.transpose(mask, (1, 0, 2))
    return array, result_array, inline_min, xline_min

train_array, result_array, inline_min, xline_min = read_images_data(TRAIN_IMAGES_PATH)
pad_size = 5
batches = []
results = []
batches_validate = []
results_validate = []
for i in range(train_array.shape[0]):
    print(i)
    for x in range(train_array.shape[1]):

        i_begin = max(i - pad_size, 0)
        i_end = min(i + pad_size, train_array.shape[0])
        current_I_begin = abs(min(i - 5, 0))
        current_I_end = 2 * pad_size + 1 - abs(min(train_array.shape[0] - i - pad_size - 1, 0))

        x_begin = max(x - pad_size, 0)
        x_end = min(x + pad_size, train_array.shape[1])
        current_X_begin = abs(min(x - 5, 0))
        current_X_end = 2 * pad_size + 1 - abs(min(train_array.shape[1] - x - pad_size - 1, 0))

        arr = np.zeros((2 * pad_size + 1, 2 * pad_size + 1, 384, 2))
        arr[current_I_begin:current_I_end, current_X_begin:current_X_end, :, :] = train_array[i_begin:i_end+1, x_begin:x_end+1, :, :]
        res = result_array[i, x]
        print("RES:", res.shape)
        if i < 30 and x < 30:
            batches_validate.append(arr)
            results_validate.append(res)
        else:
            batches.append(arr)
            results.append(res)
        break

print(len(batches))
input = tf.placeholder(tf.float32, (None, 11, 11, 384, 2), name='input')
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

batch_size = 5
batches_count = len(batches) // batch_size
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(100000):
        print("Epoch:", epoch)
        for i in range(batches_count-1):
            batch = batches[i * batch_size: (i+1) * batch_size]
            res = results[i * batch_size: (i+1) * batch_size]
            l, _ = sess.run([loss, opt], feed_dict={input: batch, result_placeholder: res})
            print("Loss:", l)

        print("VALIDATE")
        loses = []
        for b, r in zip(batches_validate, results_validate):
            l = sess.run([loss, opt], feed_dict={input: [b], result_placeholder: [r]})
            loses.append(l)
        print("VALIDATE LOSS:", np.array(loses).mean())