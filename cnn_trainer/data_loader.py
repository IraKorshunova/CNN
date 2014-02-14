import numpy as np
import theano


class DataLoader(object):
    def __init__(self, path, test_file_num, other_file_nums):

        self.n_time_points = 1000
        self.n_channels = 18
        self.path = path

        self.test_set = self._load(test_file_num, shuffle=False)

        sets = self._load(other_file_nums, shuffle=True)

        # sets_x_pos = sets[0][np.where(sets[1] == 1)]
        # sets_x_neg = sets[0][np.where(sets[1] == 0)]
        #
        # train_size_pos = sets_x_pos.shape[0]*0.8
        # train_size_neg = sets_x_neg.shape[0]*0.8
        #
        # train_set_x = np.vstack((sets_x_pos[:train_size_pos], sets_x_neg[:train_size_neg]))
        # valid_set_x = np.vstack((sets_x_pos[train_size_pos:], sets_x_neg[train_size_neg:]))
        #
        #
        # print sets_x_pos.shape

        train_size = sets[0].shape[0] * 0.8
        self.train_set = sets[0][:train_size], sets[1][:train_size]
        self.valid_set = sets[0][train_size:], sets[1][train_size:]

        self._print_stats()


    def _print_stats(self):
        print '======== dataset'
        print 'train:', self.train_set[0].shape
        print 'train number of seizures:', sum(self.train_set[1])

        print 'test:', self.test_set[0].shape
        print 'test number of seizures:', sum(self.test_set[1])

        if self.valid_set is not None:
            print 'valid:', self.valid_set[0].shape
            print 'valid number of seizures:', sum(self.valid_set[1])


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

    def get_datasets(self):
        datasets = {'train': self.train_set, 'valid': self.valid_set, 'test': self.test_set}
        return datasets