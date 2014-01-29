import glob
import re
import numpy as np
from data_loader import DataLoader
from trainer import Trainer

path = '../data/data8_npy/'
files = glob.glob(path + 'X_*.npy')
files = [f.split('/')[-1] for f in files]
p = re.compile('\d+')
file_nums = [p.findall(f)[0] for f in files]
file_nums = np.asarray(file_nums, dtype='int32')
print file_nums

test_files = [file_nums[3]]

for i in test_files:
    data_loader = DataLoader(path=path, test_file_num=i, other_file_nums=file_nums[file_nums != i], shared=False)
    datasets = data_loader.get_datasets()
    Trainer(datasets).evaluate_lenet()
