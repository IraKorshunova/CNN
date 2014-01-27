import numpy as np
import scipy.signal
import cPickle
import glob
import os

win_len = 1000 
ignore_after_seizure = win_len * 3 * 2
rate = 5

def filter_and_subsample (path_in, path_out):
    files = glob.glob(path_in)
    n_files = len(files)
        
    all_x = None
    
    for i in range(n_files):
        f_in = open(files[i], 'rb')    
        x, y = cPickle.load(f_in)
        
        # filter
        b, a = scipy.signal.butter(2, (0.5 + np.array([0, 24])) / (256 / 2), 'band')
        x = scipy.signal.lfilter(b, a, x, axis=0)
        
        # subsample
        x = x[::rate, ]
        y = y[::rate, ]
               
        data = x, y
        f_out = open(path_out + os.path.basename(f_in.name), 'wb')
        cPickle.dump(data, f_out, -1)
        
        f_in.close()
        f_out.close()
        
        if i == 0:
            all_x = x
        else:
            all_x = np.concatenate((all_x, x), axis=0)
    
    return np.mean(all_x, 0), np.std(all_x, 0)
       

def normalize(path, mean, std):
    files = glob.glob(path)
    n_files = len(files)
    
    for i in range(n_files):
        f = open(files[i], 'rb')    
        x, y = cPickle.load(f)
        f.close()
        
        x = (x - mean) / std
               
        f = open(files[i], 'wb') 
        cPickle.dump((x, y), f, -1)
        f.close()
        

def check(path):
    files = glob.glob(path)
    n_files = len(files)
    all_x = None
    
    for i in range(n_files):
        f = open(files[i], 'rb')    
        x, y = cPickle.load(f)
        
        if(i == 0):
            all_x = x
        else:
            all_x = np.concatenate((all_x, x), axis=0)
    
    print np.mean(all_x, 0)
    print np.std(all_x, 0)
    
    
def get_begin_end(y):
    idx = np.where(y == 1)[0]
        
    if len(idx) == 0:
        return []
    
    idx_offset = [i for i in idx[1:]]
    idx_offset =  np.append(idx_offset, idx[-1]+1)
    diff = (idx_offset - idx) > 1
    
    if True in diff:
        group_idx = np.where(diff == 1)[0]
        be = np.array(np.split(idx, group_idx+1))
    else:
        be = np.array([[idx[0], idx[-1]]])
    
    return be
        

def preprocess(path):
    files = glob.glob(path)
    n_files = len(files)
    
    for i in range(n_files):
    
        f = open(files[i], 'rb')    
        X, Y = cPickle.load(f)
        f.close()
            
        x, y = None, None
        be = get_begin_end(Y)
        if len(be) == 1:
            x, y = convert_data_cnn(X[:be[0, 0] - win_len, :])
            x2, y2 = convert_data_cnn(X[be[0, 0] - win_len:be[0, 1], :], y=1)
            x = np.concatenate((x, x2))
            y = np.concatenate((y, y2))
            x2, y2 = convert_data_cnn(X[be[0, 1] + ignore_after_seizure:, :], y=0)
            x = np.concatenate((x, x2))
            y = np.concatenate((y, y2))
        elif len(be) > 1:
            print 'NOT IMPLEMENTEND YET !!!'
        else:
            x, y = convert_data_cnn(X)
            
        f = open(files[i], 'wb') 
        cPickle.dump((x, y), f, -1)
        f.close()
        
            
def convert_data_cnn(x, y=0):
    ns = len(x) / win_len * win_len
    print x.shape, reshape_data_cnn(x[:ns, :]).shape, reshape_data_cnn(x[win_len / 2:ns - win_len / 2, :]).shape
    data = np.concatenate((reshape_data_cnn(x[:ns, :]), reshape_data_cnn(x[win_len / 2:ns - win_len / 2, :])))
    outp = y * np.ones((len(data), 1))
    return data, outp
    
def reshape_data_cnn(x):
    n = len(x) / win_len
    return np.transpose(np.reshape(x, (n, -1, x.shape[1])), (0, 2, 1))

def convert_to_npy(path_in, path_out):
    files = glob.glob(path_in)
    n_files = len(files)
    
    for i in range(n_files):
        f = open(files[i], 'rb')    
        x, y = cPickle.load(f)
        f.close()
        
        np.save(path_out+ 'X_'+str(i), x)
        np.save(path_out + 'Y_'+str(i), y)
       
    
    
if __name__ == "__main__":
    mean, std = filter_and_subsample('../data/data17/*.pickle','../data/data17_processed/')
    normalize('../data/data17_processed/*.pickle', mean, std)
    check('../data/data17_processed/*.pickle')
    preprocess('../data/data17_processed/*.pickle')
    convert_to_npy('../data/data17_processed/*.pickle', '../data/data17_npy/')