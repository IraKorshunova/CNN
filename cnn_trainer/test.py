import glob
import re
import numpy as np
from theano import config
from cnn.conv_net import ConvNet
from datasets_loader import DatasetsLoader


config.floatX = 'float32'
#config.profile = True

patient = '08'
init_learning_rate = 0.05
max_iters = 100000
max_fails = 100
improvement_threshold = 0.99
recept_width = [32, 32]
pool_width = [5, 5]
nkerns = [10, 20, 120]
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

#path = '/mnt/storage/usr/ikorshun/data/data08_npy/'
path = '../data/data' + patient + '_npy/'
files = glob.glob(path + 'X_*.npy')
files = [f.split('/')[-1] for f in files]
p = re.compile('\d+')
file_nums = [p.findall(f)[0] for f in files]
file_nums = np.asarray(file_nums, dtype='int32')

rng = np.random.RandomState(424242)
for i in file_nums:
    print 'test', i
    test_set = DatasetsLoader.load(path, i)
    sets = DatasetsLoader.get_train_valid_set(path, file_nums[file_nums != i], rng)
    train_set = sets['train']
    valid_set = sets['valid']

    cnn = ConvNet(nkerns, recept_width, pool_width, dropout_prob, batch_size)
    opt_iters = cnn.validate(train_set, valid_set, init_learning_rate, max_iters, validation_frequency, max_fails,
                             improvement_threshold)
    cnn = ConvNet(nkerns, recept_width, pool_width, dropout_prob, batch_size)
    train_set = np.concatenate((train_set[0], valid_set[0])), np.concatenate((train_set[1], valid_set[1]))
    cnn.test(train_set, test_set, init_learning_rate, init_learning_rate / max_iters, opt_iters)
    break