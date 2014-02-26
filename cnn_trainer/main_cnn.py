import glob
import re
import numpy as np
from theano import config
from cnn.conv_net import ConvNet
from datasets_loader import DatasetsLoader, load


config.floatX = 'float32'
#config.profile = True

patient = '8'
init_learning_rate = 0.05
max_epochs = 20
recept_width = [32, 32]
pool_width = [5, 5]
nkerns = [6, 16, 120]
dropout_prob = 0.5
improvement_threshold = 0.98
valids_per_epoch = 8
max_fails = valids_per_epoch*3
lr_decays = [1e-06, 2e-06, 0.5e-06]

print '======== parameters'
print 'patient', patient
print 'max_epochs: ', max_epochs
print 'init_learning_rate', init_learning_rate
print 'nkerns: ', nkerns
print 'receptive width: ', recept_width
print 'pool_width: ', pool_width
print 'dropout_prob: ', dropout_prob
print 'max_fails', max_fails
print 'improvement_threshold', improvement_threshold
print 'valids_rate', valids_per_epoch
print 'learning_weight_decay', lr_decays

#path = '/mnt/storage/usr/ikorshun/EEG/data8_npy/'
path = '../data/data' + patient + '_npy/'
files = glob.glob(path + 'X_*.npy')
files = [f.split('/')[-1] for f in files]
p = re.compile('\d+')
file_nums = [p.findall(f)[0] for f in files]
file_nums = np.asarray(file_nums, dtype='int32')
print file_nums

for t in file_nums:
    test_set = np.load(path + 'X_' + str(t) + ".npy"), np.load(path + 'Y_' + str(t) + ".npy")
    datasets_iterator = DatasetsLoader(path, file_nums[file_nums != t])

    valid_errors = []
    for learning_rate_decay in lr_decays:
        print 'learning_weight_decay', learning_rate_decay
        sum_err = 0
        # 3-fold-cross-validation
        for (train_set, valid_set) in datasets_iterator:
            cnn = ConvNet(nkerns, recept_width, pool_width, dropout_prob)
            err = cnn.validate(train_set, valid_set, init_learning_rate, learning_rate_decay, max_epochs,
                               max_fails, improvement_threshold, valids_per_epoch)
            sum_err += err

        valid_errors = np.append(valid_errors, sum_err)

    # test with optimal learning rate decay
    opt_decay = lr_decays[np.argmin(valid_errors)]
    conv_net = ConvNet(nkerns, recept_width, pool_width, dropout_prob)
    train_set = load(path, file_nums[file_nums != t], True)
    conv_net.test(train_set, test_set, init_learning_rate, opt_decay)


#test_files = np.asarray([3, 18, 4], dtype='int32') #for patient 17
#test_files = np.asarray([18, 12, 0,8,11], dtype='int32') #for patient 8

# ts = np.asarray([18], dtype='int32')
# data_loader = DataLoader(path=path, test_file_num=ts, other_file_nums=file_nums[file_nums != ts])
# datasets = data_loader.get_datasets()
# max_iters = ConvNet(nkerns=nkerns, recept_width=recept_width, pool_width=pool_width,
#                     dropout_prob=dropout_prob).valid_test(
#     datasets, learning_rate, max_epochs, max_fails, improvement_threshold, valids_per_epoch)
#
# # merge train and validation sets
# train_set = np.vstack((datasets['train'][0], datasets['valid'][0])), np.concatenate(
#     (datasets['train'][1], datasets['valid'][1]))
# test_set = datasets['test']
#
# # init conv_net, train and test it
# conv_net = ConvNet(nkerns=nkerns, recept_width=recept_width, pool_width=pool_width, dropout_prob=dropout_prob)
# conv_net.test(train_set=train_set, test_set=test_set, learning_rate, 40000)

