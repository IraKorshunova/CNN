import numpy as np
from datasets_loader import DatasetsLoader


class DatasetsIterator(object):
    def __init__(self, path, file_numbers, n_folds):
        self.n_time_points = 1000
        self.n_channels = 18
        self.path = path
        self.file_numbers = file_numbers
        self.n_valid_files = np.ceil(len(file_numbers) / n_folds)
        self.index = 0
        self.max_index = len(file_numbers) - 1

    def next(self):
        if self.index >= self.max_index:
            self.__restart()
            raise StopIteration
        else:
            valid_files = self.file_numbers[self.index:self.index + self.n_valid_files]
            train_files = np.concatenate(
                (self.file_numbers[:self.index], self.file_numbers[self.index + self.n_valid_files:]))
            print 'valid', valid_files
            print 'train', train_files

            train_set = DatasetsLoader.load(self.path, train_files, True)
            valid_set = DatasetsLoader.load(self.path, valid_files, False)

            self.index += self.n_valid_files

        return train_set, valid_set

    def __iter__(self):
        return self

    def __restart(self):
        self.index = 0