import os
import pickle

import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from RleUtils import rle2mask

TRAIN_IMAGES_PATH = "/media/andrey/ssdbig1/data/rosneft/train"
masks_pd = pd.read_csv("/media/andrey/ssdbig1/data/rosneft/seismic_challenge_train(masks).csv")


inline_min = 10000
inline_max = 0
xline_min = 10000
xline_max = 0
for file_name in os.listdir(TRAIN_IMAGES_PATH):
    type = file_name[:file_name.index("_")]
    index = int(file_name[file_name.index("_") + 1:file_name.index(".")])
    if type == "xline":
        xline_min = min(xline_min, index)
        xline_max = max(xline_max, index)
    if type == "inline":
        inline_min = min(inline_min, index)
        inline_max = max(inline_max, index)

array = np.zeros((inline_max - inline_min + 1, xline_max - xline_min + 1, 384, 1))
result_array = np.zeros((inline_max - inline_min + 1, xline_max - xline_min + 1, 384, 8))
for file_name in os.listdir(TRAIN_IMAGES_PATH):
    type = file_name[:file_name.index("_")]
    index = int(file_name[file_name.index("_") + 1:file_name.index(".")])
    if type == "xline": continue
    print(file_name)
    img = cv2.imread(os.path.join(TRAIN_IMAGES_PATH, file_name), cv2.IMREAD_GRAYSCALE)
    mask = np.zeros((img.shape[0], img.shape[1], 8))
    for i, rle in enumerate(masks_pd[masks_pd['ImageId'] == file_name]['EncodedPixels']):
        mask[:, :, i+1] = np.array(rle2mask(rle, shape=(img.shape[1], img.shape[0])), dtype=np.float)

    #fig, ax = plt.subplots(1)
    #ax.imshow(mask[:, :, 3:6] + mask[:, :, 0:3])
    #plt.show()

    #if type == "xline":
    #    array[:, index - xline_min, :, 1] = (np.transpose(img, (1, 0)) - 127.5) / 127.5
    #    result_array[:, index - xline_min, :] = np.transpose(mask, (1, 0, 2))
    if type == "inline":
        array[index - inline_min:, :, :, 0] = (np.transpose(img, (1, 0)) - 127.5) / 127.5
        result_array[index - inline_min:, :, :] = np.transpose(mask, (1, 0, 2))

with open("/media/andrey/ssdbig1/data/rosneft/inline3d.pkl", 'wb') as f:
    pickle.dump(array, f, protocol=4)
with open("/media/andrey/ssdbig1/data/rosneft/inline3d_result.pkl", 'wb') as f:
    pickle.dump(result_array, f, protocol=4)