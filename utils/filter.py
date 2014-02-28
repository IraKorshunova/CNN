import numpy as np
import scipy.signal
import cPickle
import glob
import shutil
import os

win_len = 1000
ignore_after_seizure = win_len * 3 * 2
rate = 5


def filter_subsample_normalize(path_in, path_out):
    files = sorted(glob.glob(path_in + '/*.pickle'))
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

        f_out = open(path_out +'/'+ os.path.basename(f_in.name), 'wb')
        cPickle.dump((x,y), f_out, -1)
        f_out.close()

        if i == 0:
            all_x = x
        else:
            all_x = np.concatenate((all_x, x), axis=0)

    mean = np.mean(all_x, 0)
    std = np.std(all_x, 0)

    files = sorted(glob.glob(path_out+'/*.pickle'))
    for i in range(n_files):
        f = open(files[i], 'rb')
        x, y = cPickle.load(f)
        f.close()

        x = (x - mean) / std

        f = open(files[i], 'wb')
        cPickle.dump((x, y), f, -1)
        f.close()

def check(path):
    files = glob.glob(path)
    n_files = len(files)
    all_x = None

    for i in range(n_files):
        f = open(files[i], 'rb')
        x, y = cPickle.load(f)

        if (i == 0):
            all_x = x
        else:
            all_x = np.concatenate((all_x, x), axis=0)

    print all_x.shape
    print np.mean(all_x, 0)
    print np.std(all_x, 0)


def get_begin_end(x):
    be = np.where(np.logical_xor(x[:-1], x[1:]))[0] + 1
    if x[0] > 0:
        be = np.concatenate(([0], be))
    if x[-1] > 0:
        be = np.concatenate((be, [len(x)]))
    return np.reshape(be, (len(be) / 2, 2))


def preprocess(path_in, path_out):
    files = glob.glob(path_in+'/*.pickle')
    n_files = len(files)

    for i in range(n_files):
        print '------------------------'
        print 'file:', i
        if i == 19:
            print 'lll'
        f = open(files[i], 'rb')
        X, Y = cPickle.load(f)
        f.close()

        x, y = None, None
        be = get_begin_end(Y)
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
            print 'NOT IMPLEMENTEND YET !!!'
            print be

            x, y = convert_data_cnn(X[:be[0, 0] - win_len, :])  # first seizure
            for i in range(len(be)):
                x2, y2 = convert_data_cnn(X[be[i, 0] - win_len:be[i, 1], :], y=1)
                x = np.concatenate((x, x2))
                y = np.concatenate((y, y2))

                if i == len(be) - 1: # last seizure
                    x2, y2 = convert_data_cnn(X[be[i, 1] + ignore_after_seizure:, :], y=0)
                else:
                    x2, y2 = convert_data_cnn(X[be[i, 1] + ignore_after_seizure: be[i + 1, 0] - win_len, :], y=0)

                x = np.concatenate((x, x2))
                y = np.concatenate((y, y2))

        else:
            x, y = convert_data_cnn(X)

        x = np.transpose(x, (0, 2, 1))
        print 'output_shape', x.shape
        np.save(path_out + 'X_' + str(i), x)
        np.save(path_out + 'Y_' + str(i), y)


def convert_data_cnn(x, y=0):
    if len(x) < win_len:
        return np.ones((0, 18, win_len)), np.ones((0, 1))
    else:
        ns = len(x) / win_len * win_len
        print ns
        print 'xshape', x.shape

        if len(x) > 2 * win_len:
            x1 = reshape_data_cnn(x[:ns, :])
            x2 = reshape_data_cnn(x[win_len / 2:ns - win_len / 2, :])
            print x.shape, x1.shape, x2.shape
            data_x = np.zeros((x1.shape[0]+x2.shape[0], x1.shape[1], x1.shape[2]))
            data_x[0::2] = x1
            data_x[1::2] = x2
        else:
            data_x = reshape_data_cnn(x[:ns, :])

        data_y = y * np.ones((len(data_x), 1))
        return data_x, data_y

def reshape_data_cnn(x):
    n = len(x) / win_len
    return np.transpose(np.reshape(x, (n, -1, x.shape[1])), (0, 2, 1))


if __name__ == "__main__":
    patient = '24'
    os.mkdir('../data/data'+patient+'_processed')
    os.mkdir('../data/data'+patient+'_npy')
    filter_subsample_normalize('../data/chb'+patient, '../data/data'+patient+'_processed')
    check('../data/data'+patient+'_processed/*.pickle')
    preprocess('../data/data'+patient+'_processed', '../data/data'+patient+'_npy/')
    shutil.rmtree('../data/data'+patient+'_processed')