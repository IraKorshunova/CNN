import numpy as np
import cPickle 
import glob
import re


n_time_points = 1000
n_channels = 18

files =  glob.glob("./data/more/train/X_*.npy")
p = re.compile('\d+')
file_nums = [p.findall(f)[0] for f in files]

train_set_x = 0
train_set_y = 0
for i in  file_nums:
    print i
    x = np.load("./data/more/train/X_" + i + ".npy")
    x = np.reshape(x, (-1, n_time_points * n_channels), order = 'F')  # by columns
    y = np.load("./data/more/train/Y_" + i + ".npy")
    y = np.squeeze(np.asarray(y))
    if i == file_nums[0]:
        train_set_x = x
        train_set_y = y
    else:
        train_set_x = np.concatenate((train_set_x, x), axis=0)
        train_set_y = np.concatenate((train_set_y, y), axis=0)
      
train_set = train_set_x, train_set_y
print train_set_x.shape
print train_set_y.shape


files =  glob.glob("./data/more/validation/X_*.npy")
p = re.compile('\d+')
file_nums = [p.findall(f)[0] for f in files]
  
valid_set_x = 0
valid_set_y = 0
valid_batch_sizes  = []
for i in file_nums:
    print i
    x = np.load("./data/more/validation/X_" + i + ".npy")
    x = np.reshape(x, (-1, n_time_points * n_channels), order = 'F')
    y = np.load("./data/more/validation/Y_" + i + ".npy")
    y = np.squeeze(np.asarray(y))
    valid_batch_sizes.append(x.shape[0])
    if i == file_nums[0]:
        valid_set_x= x
        valid_set_y = y
    else:
        valid_set_x = np.concatenate((valid_set_x, x), axis=0) 
        valid_set_y = np.concatenate((valid_set_y, y), axis=0)
      
valid_set = valid_set_x, valid_set_y
print valid_set_x.shape
print valid_set_y.shape
  
sets = train_set, valid_set
  
f = open('more_data.pkl', 'wb')
data = cPickle.dump(sets, f, -1)
f.close()

valid_batch_sizes = np.array(valid_batch_sizes)
f = open('valid_batch_sizes.pkl', 'wb')
data = cPickle.dump(valid_batch_sizes, f, -1)
f.close()
 
 


