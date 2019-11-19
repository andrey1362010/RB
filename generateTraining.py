import pickle
import random

import numpy as np


class DataLoader:
        def __init__(self):
            with open("/media/andrey/ssdbig1/data/rosneft/inline3d.pkl", 'rb') as f:
                self.train_array = pickle.load(f)
            with open("/media/andrey/ssdbig1/data/rosneft/inline3d_result.pkl", 'rb') as f:
                self.result_array = pickle.load(f)

        def get(self, batch_size=5, pad_size=10):
            sz = 1 + 2 * pad_size
            batches = np.zeros((batch_size, sz, sz, 384, 1))
            results = np.zeros((batch_size, 384, 8))
            for count in range(batch_size):
                i = random.randrange(self.train_array.shape[0])
                j = random.randrange(self.train_array.shape[1])
                results[count] = self.result_array[i, j]

                i_begin = max(i - pad_size, 0)
                j_begin = max(j - pad_size, 0)
                i_end = min(i + pad_size, self.train_array.shape[0])
                j_end = min(j + pad_size, self.train_array.shape[1])

                current_I_begin = abs(min(i - pad_size, 0))
                current_I_end = 2 * pad_size + 1 - abs(min(self.train_array.shape[0] - i - pad_size - 1, 0))
                current_X_begin = abs(min(j - pad_size, 0))
                current_X_end = 2 * pad_size + 1 - abs(min(self.train_array.shape[1] - j - pad_size - 1, 0))

                batches[count, current_I_begin:current_I_end, current_X_begin:current_X_end] = self.train_array[i_begin:i_end+1, j_begin:j_end+1, :, :]
            return batches, results
