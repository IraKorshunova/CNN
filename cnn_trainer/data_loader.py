import numpy as np
import theano


class DataLoader(object):
    def __init__(self, path, test_file_num, other_file_nums, shared=True):

        self.n_time_points = 1000
        self.n_channels = 18
        self.path = path

        self.test_mode = False if test_file_num is None else True

        if self.test_mode:
            self.test_set = self._load(test_file_num, shuffle=False)

        sets = self._load(other_file_nums, shuffle=True)
        train_size = sets[0].shape[0] * 0.8

        self.train_set = sets[0][:train_size], sets[1][:train_size]
        self.valid_set = sets[0][train_size:], sets[1][train_size:]

        self._print_stats()

        if shared:
            self.test_set = self._shared_dataset(self.test_set)
            self.valid_set = self._shared_dataset(self.valid_set)
            if self.test_mode:
                self.train_set = self._shared_dataset(self.train_set)

    def _print_stats(self):
        print '======== dataset'
        print 'train:', self.train_set[0].shape
        print 'train number of seizures:', sum(self.train_set[1])

        print 'valid:', self.valid_set[0].shape
        print 'valid number of seizures:', sum(self.valid_set[1])

        if self.test_mode:
            print 'test:', self.test_set[0].shape
            print 'test number of seizures:', sum(self.test_set[1])

    def _load(self, file_numbers, shuffle):
        if file_numbers.shape == ():
            file_numbers = np.array([file_numbers], dtype='int32')
        x = 0
        y = 0
        for i in file_numbers:
            x_temp = np.load(self.path + 'X_' + str(i) + ".npy")
            x_temp = np.reshape(x_temp, (-1, self.n_time_points * self.n_channels), order='F')  # by columns
            y_temp = np.load(self.path + 'Y_' + str(i) + ".npy")
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

    def _shared_dataset(self, data, borrow=True):
        x, y = data
        shared_x = theano.shared(np.asarray(x, dtype='float32'), borrow=borrow)
        shared_y = theano.shared(np.asarray(y, dtype='int8'), borrow=borrow)
        return shared_x, shared_y

    def get_datasets(self):
        if self.test_mode:
            return self.train_set, self.valid_set, self.test_set
        else:
            return self.train_set, self.valid_set
