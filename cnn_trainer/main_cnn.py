import glob
import re
import numpy as np
from theano import config
from data_loader import DataLoader
from cnn.conv_net import ConvNet


config.floatX = 'float32'
#config.profile = True

patient = '8'
learning_rate = 0.05
max_epochs = 15
recept_width = [32, 32]
pool_width = [5, 5]
nkerns = [6, 16, 120]
dropout_prob = 0.5
max_fails = 12
improvement_threshold = 0.98
valids_per_epoch = 4

print '======== parameters'
print 'patient', patient
print 'max_epochs: ', max_epochs
print 'learning_rate', learning_rate
print 'nkerns: ', nkerns
print 'receptive width: ', recept_width
print 'pool_width: ', pool_width
print 'dropout_prob: ', dropout_prob
print 'max_fails', max_fails
print 'improvement_threshold', improvement_threshold
print 'valids_rate', valids_per_epoch

#path = '/mnt/storage/usr/ikorshun/EEG/data8_npy/'
path = '../data/data' + patient + '_npy/'
files = glob.glob(path + 'X_*.npy')
files = [f.split('/')[-1] for f in files]
p = re.compile('\d+')
file_nums = [p.findall(f)[0] for f in files]
file_nums = np.asarray(file_nums, dtype='int32')
print file_nums

#test_files = np.asarray([3, 18, 4], dtype='int32') #for patient 17
#test_files = np.asarray([18, 12, 0,8,11], dtype='int32') #for patient 8

ts = np.asarray([18], dtype='int32')
data_loader = DataLoader(path=path, test_file_num=ts, other_file_nums=file_nums[file_nums != ts])
datasets = data_loader.get_datasets()
max_iters = ConvNet(nkerns=nkerns, recept_width=recept_width, pool_width=pool_width,
                      dropout_prob=dropout_prob).valid_test(
      datasets, learning_rate, max_epochs, max_fails, improvement_threshold, valids_per_epoch)

ConvNet(nkerns=nkerns, recept_width=recept_width, pool_width=pool_width, dropout_prob=dropout_prob).test(
    datasets, learning_rate, 40000)

