import numpy as np
np.set_printoptions(threshold=np.nan)
import glob
import re

x0 = np.load('../data/data08_npy/X_0.npy')
y0 = np.load('../data/data08_npy/Y_0.npy')

print x0[0:10,1:5,0]
print x0.shape
print np.sum(y0)

x0 = np.load('../data/data88_npy/X_0.npy')
y0 = np.load('../data/data88_npy/Y_0.npy')

print x0[0:10,1:5,0]
print x0.shape
print np.sum(y0)

#'/'.join(os.getcwd().split('/')[:-1]) +

path = '../data/data8_npy/'
files = glob.glob(path + 'X_*.npy')
files = [f.split('/')[-1] for f in files]
p = re.compile('\d+')
file_nums = [p.findall(f)[0] for f in files]
file_nums = np.asarray(file_nums, dtype='int32')
print file_nums

for i in file_nums:
    s = np.sum(np.load('../data/data8_npy/Y_' + str(i) +'.npy'))
    if s > 0:
        print i, s
