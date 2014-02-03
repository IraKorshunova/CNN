import numpy as np


class DataIterator(object):

    def __init__(self, dataset):
        self.x, self.y = dataset
        self.index = 0
        self.max_index = self.x.shape[0] - 1

    def __iter__(self):
        return self

    def next(self):
        if self.index > self.max_index:
            self.__restart()
            raise StopIteration
        else:
            x = np.transpose(self.x[self.index, :, None])
            y = np.array([self.y[self.index,]])
            self.index += 1
        return x,y

    def __restart(self):
        self.index = 0


    def get_number_of_batches(self):
        return  self.max_index + 1