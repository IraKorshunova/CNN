import numpy as np


class TrainSetIterator(object):

    def __init__(self, dataset, batch_size):
        x, y = dataset
        np.random.seed(42)
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        self.x = x[idx, :]
        self.y = y[idx]
        self.index = 0
        self.batch_size = batch_size
        self.size = self.x.shape[0]


    def __iter__(self):
        return self

    def next(self):
        begin = self.index
        end = self.index + self.batch_size
        if end <= self.size:
            x = self.x[begin:end, :]
            y = self.y[begin:end]
            self.index += self.batch_size
            return x, y
        else:
             self.__restart()
             raise StopIteration


    def __restart(self):
        np.random.seed(42)
        idx = np.arange(self.size)
        np.random.shuffle(idx)
        self.x = self.x[idx, :]
        self.y = self.y[idx]
        self.index = 0

    def get_number_of_batches(self):
        return  self.size / self.batch_size