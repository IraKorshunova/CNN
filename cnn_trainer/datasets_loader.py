import numpy as np


def get_train_valid_set(path, file_numbers, rng):
    train_valid_files = get_train_valid_files(path, file_numbers, rng)
    train_set = load(path, train_valid_files['train'])
    valid_set = load(path, train_valid_files['valid'])
    return {'train': train_set, 'valid': valid_set}


def get_train_valid_files(path, file_numbers, rng):
    seizure_files = []
    nonseizure_files = []

    for i in file_numbers:
        y = np.load(path + 'Y_' + str(i) + ".npy")
        if np.sum(y) > 0:
            seizure_files.append(i)
        else:
            nonseizure_files.append(i)

    valid_files = list(rng.choice(seizure_files, max(1, np.rint(len(seizure_files) / 5.0)), False))
    ns_valid_files = rng.choice(nonseizure_files, max(1, np.rint(len(nonseizure_files) / 5.0)), False)
    for i in ns_valid_files:
        valid_files.append(i)

    train_files = list(set(file_numbers) - set(valid_files))

    return {'train': train_files, 'valid': valid_files}


def get_begin_end(x):
    be = np.where(np.logical_xor(x[:-1], x[1:]))[0] + 1
    if x[0] > 0:
        be = np.concatenate(([0], be))
    if x[-1] > 0:
        be = np.concatenate((be, [len(x)]))
    return np.reshape(be, (len(be) / 2, 2))


def load(path, file_numbers):
    x = 0
    y = 0
    for i in file_numbers:
        x_temp = np.load(path + 'X_' + str(i) + ".npy")
        #x_temp = np.reshape(x_temp, (-1, n_time_points * n_channels), order='F')  # by columns
        y_temp = np.load(path + 'Y_' + str(i) + ".npy")
        y_temp = np.squeeze(y_temp)
        if i == file_numbers[0]:
            x = x_temp
            y = y_temp
        else:
            x = np.concatenate((x, x_temp), axis=0)
            y = np.concatenate((y, y_temp), axis=0)

    #y[np.where(y == 0)[0]] = -1
    x = np.float32(x)
    y = np.int8(y)

    return x, y