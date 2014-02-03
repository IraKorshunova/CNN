import glob
import re
import sys
import os
import numpy as np
from data_loader import DataLoader
from trainer import Trainer
from theano import config
import ConfigParser
import json

config.profile = True
config.floatX = 'float32'


config_file = ConfigParser.RawConfigParser()
config_file.read('config.cfg')

patient = config_file.get('T', 'patient')
learning_rate = config_file.getfloat('T', 'learning_rate')
n_epochs = config_file.getint('T', 'n_epochs')
nkerns = json.loads(config_file.get('T', 'nkerns'))
pool_width = json.loads(config_file.get('T', 'pool_width'))
receptive_width = json.loads(config_file.get('T', 'receptive_width'))
dropout_prob = config_file.getfloat('T', 'dropout_prob')

path = '/'.join(os.getcwd().split('/')[:-1]) + '/data/data' + patient + '_npy/'
files = glob.glob(path + 'X_*.npy')
files = [f.split('/')[-1] for f in files]
p = re.compile('\d+')
file_nums = [p.findall(f)[0] for f in files]
file_nums = np.asarray(file_nums, dtype='int32')
print file_nums

#test_files =np.asarray( [3,18,4], dtype='int32') #for patient 17
test_files = np.asarray([18, 12, 17, 8, 11], dtype='int32')  # for patient 8

for i in test_files:
    print '--------------- test file:', i
    data_loader = DataLoader(path=path, test_file_num=i, other_file_nums=file_nums[file_nums != i], shared=False)
    datasets = data_loader.get_datasets()
    Trainer(datasets).evaluate_lenet(learning_rate=learning_rate, n_epochs=n_epochs,
                                     nkerns=nkerns, recept_width=receptive_width, pool_width=pool_width,
                                     dropout_prob=dropout_prob)
