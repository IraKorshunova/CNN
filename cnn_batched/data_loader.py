import numpy as np
import glob
import re


class DataLoader(object):

    def __init__(self, path):

        self.path = path
        self.n_time_points = 1000
        self.n_channels = 18

        files = glob.glob(path + 'X_*.npy')
        files = [f.split('/')[-1] for f in files]
        p = re.compile('\d+')
        file_nums = [p.findall(f)[0] for f in files]
        file_nums = np.array(file_nums)
        print file_nums

        np.random.seed(42)

        test_set = self.load(['20'], shuffle=False)
        sets = self.load(file_nums[file_nums != '20'], shuffle=False)
        train_size = sets[0].shape[0]*0.8
        train_set = sets[0][:train_size], sets[1][:train_size]
        valid_set = sets[0][train_size:], sets[1][train_size:]
        self.datasets = (train_set, valid_set, test_set)

    def load(self, file_numbers, shuffle):
        x = 0
        y = 0
        for i in file_numbers:
            x_temp = np.load(self.path + 'X_' + i + '.npy')
            x_temp = np.reshape(x_temp, (-1, self.n_time_points * self.n_channels), order = 'F')  # by columns
            y_temp = np.load(self.path + 'Y_' + i + '.npy')
            y_temp = np.squeeze(np.asarray(y_temp))
            if i == file_numbers[0]:
                x = x_temp
                y = y_temp
            else:
                x = np.concatenate((x, x_temp), axis=0)
                y = np.concatenate((y, y_temp), axis=0)

        x = np.float32(x)
        y = np.int32(y)

        if shuffle:
            idx = np.arange(x.shape[0])
            np.random.shuffle(idx)
            x = x[idx, :]
            y = y[idx]

        data = x, y
        return data

    def get_datasets(self):
        return self.datasets