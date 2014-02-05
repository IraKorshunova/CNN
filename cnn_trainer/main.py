import glob
from itertools import imap
import re
import os
import numpy as np
from data_loader import DataLoader
from trainer import Trainer

#from theano import config
#config.profile = True

patient = '17'
learning_rate = 0.01
n_epochs = 20
recept_width = [32, 32]
pool_width = [5, 5]
nkerns = [6, 16, 120]
dropout_prob = 0.5
max_fails = 5
improvement_threshold = 0.05
valids_per_epoch = 2

print '======== parameters'
print 'patient', patient
print 'n_epochs: ', n_epochs
print 'learning_rate', learning_rate
print 'nkerns: ', nkerns
print 'receptive width: ', recept_width
print 'pool_width: ', pool_width
print 'dropout_prob: ', dropout_prob
print 'max_fails', max_fails
print 'improvement_threshold', improvement_threshold
print 'valids_rate', valids_per_epoch

path = '../data/data' + patient + '_npy/'
files = glob.glob(path + 'X_*.npy')
files = [f.split('/')[-1] for f in files]
p = re.compile('\d+')
file_nums = [p.findall(f)[0] for f in files]
file_nums = np.asarray(file_nums, dtype='int32')
print file_nums

test_files = np.asarray([3, 18, 4], dtype='int32') #for patient 17

for i in test_files:
    print '--------------- test file:', i
    data_loader = DataLoader(path=path, test_file_num=i, other_file_nums=file_nums[file_nums != i], shared=False)
    datasets = data_loader.get_datasets()
    Trainer(datasets).evaluate_lenet(learning_rate=learning_rate, n_epochs=n_epochs,
                                     nkerns=nkerns, recept_width=recept_width, pool_width=pool_width,
                                     dropout_prob=dropout_prob, max_fails=max_fails,
                                     improvement_threshold=improvement_threshold,
                                     valids_per_epoch=valids_per_epoch)
