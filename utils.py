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
