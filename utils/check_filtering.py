import numpy as np
np.set_printoptions(threshold=np.nan)
import glob
import re

def check_data():
    x0 = np.load('../data/data08_npy/X_29.npy')
    y0 = np.load('../data/data08_npy/Y_29.npy')

    print x0[0:10,995:1000,0]
    print x0.shape
    print np.sum(y0)

    x0 = np.load('../data/data08_1npy/X_29.npy')
    y0 = np.load('../data/data08_1npy/Y_29.npy')

    print x0[0:10,995:1000,0]
    print x0.shape
    print np.sum(y0)

    # x0 = np.load('../data/data8_npy/X_14.npy')
    # y0 = np.load('../data/data8_npy/Y_14.npy')
    #
    # print x0[0:10,1:5,0]
    # print x0.shape
    # print np.sum(y0)


def get_begin_end(x):
    be = np.where(np.logical_xor(x[:-1], x[1:]))[0] + 1
    if x[0] > 0:
        be = np.concatenate(([0], be))
    if x[-1] > 0:
        be = np.concatenate((be, [len(x)]))
    return np.reshape(be, (len(be) / 2, 2))


def check_seizures_number():
    path = '../data/data08_npy/'
    files = glob.glob(path + 'X_*.npy')
    files = [f.split('/')[-1] for f in files]
    p = re.compile('\d+')
    file_nums = [p.findall(f)[0] for f in files]
    file_nums = np.asarray(file_nums, dtype='int32')
    print file_nums

    n_seizures = 0
    avg_len = 0
    for i in file_nums:
        y = np.load('../data/data08_npy/Y_' + str(i) +'.npy')
        s = np.sum(y)
        if s > 0:
            be = get_begin_end(y)
            n_seizures += len(be)
            for j in range(len(be)):
                avg_len += (be[0,1] - be[0,0])/2+1
            print '===================file:', i
            print be
            for j in range(len(be)):
                print (be[j,1] - be[j,0])/2+ 1
            print '--------------------'
    print 'n_seizures', n_seizures
    print 'avg_len', avg_len * 10.0/n_seizures


if __name__=='__main__':
    check_data()
    check_seizures_number()