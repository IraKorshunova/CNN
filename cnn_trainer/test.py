import glob
import re
import numpy as np
from theano import config
from cnn.conv_net import ConvNet
from datasets_loader import DatasetsLoader


config.floatX = 'float32'
#config.profile = True

patient = '8'
init_learning_rate = 0.05
max_iters = 40000
recept_width = [32, 32]
pool_width = [5, 5]
nkerns = [6, 16, 120]
dropout_prob = 0.5
validation_frequency = 200
batch_size = 1

print '======== parameters'
print 'patient', patient
print 'init_learning_rate', init_learning_rate
print 'max_iters', max_iters
print 'nkerns: ', nkerns
print 'receptive width: ', recept_width
print 'pool_width: ', pool_width
print 'dropout_prob: ', dropout_prob
print 'valids_rate', validation_frequency
print 'batch_size', batch_size

#path = '/mnt/storage/usr/ikorshun/EEG/data8_npy/'
path = '../data/data' + patient + '_npy/'
files = glob.glob(path + 'X_*.npy')
files = [f.split('/')[-1] for f in files]
p = re.compile('\d+')
file_nums = [p.findall(f)[0] for f in files]
file_nums = np.asarray(file_nums, dtype='int32')

cnn = ConvNet(nkerns, recept_width, pool_width, dropout_prob, batch_size)
valid_files = np.asarray([18, 0, 3, 15], dtype='int32')
train_files = np.asarray([9, 2, 19, 12, 5, 4, 7, 8, 17, 11, 1, 6, 13, 10, 14, 16], dtype='int32')
valid_set = DatasetsLoader.load(path, valid_files)
train_set = DatasetsLoader.load(path, train_files)
err = cnn.validate(train_set, valid_set, init_learning_rate, max_iters, validation_frequency)


# for t in file_nums:
#     train = DatasetsLoader.load(path, file_nums[file_nums!=t])
#     test = DatasetsLoader.load(path, t)
#     print 'test file', t
#     cnn = ConvNet(nkerns, recept_width, pool_width, dropout_prob, batch_size)
#     err = cnn.test(train, test, init_learning_rate, max_iters)