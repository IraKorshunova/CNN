import numpy as np


class DataIterator(object):

    def __init__(self, dataset):
        self.dataset = dataset
        x, y = dataset

        self.x0 = x[np.where(y == 0)]
        self.x1 = x[np.where(y == 1)]

        self.index = 0
        self.size = self.x1.shape[0]

    def get_number_of_batches(self):
        return np.int(np.ceil(np.true_divide(self.x0.shape[0], self.size)))

    def get_next_batch(self):

        x = np.concatenate((self.x0[self.index:self.index+self.size], self.x1))
        y = np.concatenate((np.zeros(x.shape[0]-self.size),  np.ones(self.size)))

        x = np.asarray(x, 'float32')
        y = np.asarray(y, 'int32')

        self.index += self.size
        if self.index > self.x0.shape[0]:
            self.index = 0

        return x, y




