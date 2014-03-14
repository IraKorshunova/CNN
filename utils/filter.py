import numpy as np
import scipy.signal
import cPickle
import glob
import shutil
import os
import re

stride = 1  # in seconds
win_len = 1000
n_channels = 18
ignore_after_seizure = win_len * 3 * 2
rate = 5


def filter_subsample_normalize(path_in, path_out):
    files = glob.glob(path_in + '/*.pickle')
    n_files = len(files)

    all_x = None

    for i in range(n_files):
        f_in = open(files[i], 'rb')
        x, y = cPickle.load(f_in)
        f_in.close()

        # filter
        b, a = scipy.signal.butter(2, (0.5 + np.array([0, 24])) / (256 / 2), 'band')
        x = scipy.signal.lfilter(b, a, x, axis=0)

        # subsample
        x = x[::rate, ]
        y = y[::rate, ]

        f_out = open(path_out + '/' + os.path.basename(f_in.name), 'wb')
        cPickle.dump((x, y), f_out, -1)
        f_out.close()

        if i == 0:
            all_x = x
        else:
            all_x = np.concatenate((all_x, x), axis=0)

    mean = np.mean(all_x, 0)
    std = np.std(all_x, 0)

    files = glob.glob(path_out + '/*.pickle')
    for i in range(n_files):
        f = open(files[i], 'rb')
        x, y = cPickle.load(f)
        f.close()

        x = (x - mean) / std

        x = np.float32(x)
        y = np.int8(y)

        f = open(files[i], 'wb')
        cPickle.dump((x, y), f, -1)
        f.close()


def get_begin_end(x):
    be = np.where(np.logical_xor(x[:-1], x[1:]))[0] + 1
    if x[0] > 0:
        be = np.concatenate(([0], be))
    if x[-1] > 0:
        be = np.concatenate((be, [len(x)]))
    return np.reshape(be, (len(be) / 2, 2))


def preprocess(path_in, path_out):
    files = glob.glob(path_in + '/*.pickle')
    n_files = len(files)
    p = re.compile('\d+')

    for i in range(n_files):
        print '------------------------'
        f = open(files[i], 'rb')
        X, Y = cPickle.load(f)
        f.close()

        be = get_begin_end(Y)
        print '===================================================file:', os.path.basename(f.name)
        print 'input shape:', X.shape
        print 'begin_end', be

        if len(be) == 1:
            begin = be[0, 0] - win_len
            if begin > 0:
                x, y = convert_data_cnn(X[:begin, :], y=0)
                x2, y2 = convert_data_cnn(X[begin:be[0, 1], :], y=1)
                x = np.concatenate((x, x2))
                y = np.concatenate((y, y2))
            else:
                x, y = convert_data_cnn(X[be[0, 0]:be[0, 1], :], y=1)

            end = be[0, 1] + ignore_after_seizure
            if end < len(X):
                x2, y2 = convert_data_cnn(X[end:, :], y=0)
                x = np.concatenate((x, x2))
                y = np.concatenate((y, y2))

        elif len(be) > 1:
            x, y = convert_data_cnn(X[:be[0, 0] - win_len, :], y=0)  # before first seizure
            for j in range(len(be)):
                x2, y2 = convert_data_cnn(X[be[j, 0] - win_len:be[j, 1], :], y=1)
                x = np.concatenate((x, x2))
                y = np.concatenate((y, y2))

                if j == len(be) - 1: # last seizure
                    x2, y2 = convert_data_cnn(X[be[j, 1] + ignore_after_seizure:, :], y=0)
                else:
                    x2, y2 = convert_data_cnn(X[be[j, 1] + ignore_after_seizure: be[j + 1, 0] - win_len, :], y=0)

                x = np.concatenate((x, x2))
                y = np.concatenate((y, y2))
        else:
            x, y = convert_data_cnn(X)

        print 'output_shape', x.shape
        number = str(int(p.findall(os.path.basename(f.name))[1]))
        np.save(path_out + 'X_' + number, x)
        np.save(path_out + 'Y_' + number, y)
        f.close()


def convert_data_cnn(x, y=0):
    if len(x) > win_len:
        idx = range(0, len(x) - win_len, stride*100)
        data_x = np.zeros((len(idx), win_len, n_channels), dtype='float32')
        j = 0
        for i in idx:
            data_x[j] = x[i:i + win_len, :]
            j += 1
        data_y = y * np.ones((len(data_x), 1))
    else:
        data_x, data_y = np.ones((0, win_len, n_channels)), np.ones((0, 1))
    return data_x, data_y


if __name__ == "__main__":
    patient = '08'
    os.mkdir('../data/data' + patient + '_processed')
    os.mkdir('../data/data' + patient + '_npy')
    filter_subsample_normalize('../data/chb' + patient, '../data/data' + patient + '_processed')
    preprocess('../data/data' + patient + '_processed', '../data/data' + patient + '_npy/')
    shutil.rmtree('../data/data' + patient + '_processed')