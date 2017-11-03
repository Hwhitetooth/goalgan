import numpy as np
from collections import Iterator

class Dataset(Iterator):
    def __init__(self, data_map, batch_size):
        self.data_map = data_map
        self.size = len(next(iter(data_map.values())))
        self.batch_size = batch_size
        self.shuffle()

    def shuffle(self):
        order = np.random.permutation(self.size)
        for key in self.data_map:
            self.data_map[key] = self.data_map[key][order]
        self.cur = 0

    def __next__(self):
        if self.cur == self.size:
            self.shuffle()
            raise StopIteration
        batch_size = min(self.batch_size, self.size - self.cur)
        self.cur += batch_size
        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][self.cur - batch_size: self.cur]
        return data_map

# TODO: This input normalization does NOT work.
class MovingMeanVar(object):
    def __init__(self, shape):
        self.sum = np.zeros(shape)
        self.sumsq = np.zeros(shape) + 1E-2
        self.mean = np.zeros(shape)
        self.std = np.zeros(shape)
        self.n = 1E-2

    def update(self, x):
        self.n += x.shape[0]
        self.sum += np.sum(x, axis = 0) 
        self.sumsq += np.sum(np.square(x), axis = 0)
        self.mean = self.sum / self.n
        self.std = np.sqrt(np.maximum(self.sumsq / self.n - np.square(self.mean), 1E-2))
