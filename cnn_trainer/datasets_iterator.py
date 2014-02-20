import numpy as np


class DatasetsIterator(object):
    def __init__(self, path, file_numbers):
        self.n_time_points = 1000
        self.n_channels = 18
        self.path = path
        self.file_numbers = file_numbers
        self.n_valid_files = np.ceil(len(file_numbers) / 3)
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

            train_set = load(self.path, train_files, True)
            valid_set = load(self.path, valid_files, False)

            self.index += self.n_valid_files

        return train_set, valid_set

    def __iter__(self):
        return self

    def __restart(self):
        self.index = 0


def load(path, file_numbers, shuffle,  n_time_points=1000, n_channels=18):
    if file_numbers.shape == ():
        file_numbers = np.array([file_numbers], dtype='int32')
    x = 0
    y = 0
    for i in file_numbers:
        x_temp = np.load(path + 'X_' + str(i) + ".npy")
        x_temp = np.reshape(x_temp, (-1, n_time_points * n_channels), order='F')  # by columns
        y_temp = np.load(path + 'Y_' + str(i) + ".npy")
        y_temp = np.squeeze(y_temp)
        if i == file_numbers[0]:
            x = x_temp
            y = y_temp
        else:
            x = np.concatenate((x, x_temp), axis=0)
            y = np.concatenate((y, y_temp), axis=0)

    if shuffle:
        np.random.seed(42)
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        x = x[idx, :]
        y = y[idx]

    x = np.float32(x)
    y = np.int8(y)

    return x, y