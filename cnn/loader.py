import cPickle
import numpy as np
import theano

def shared_dataset(data_xy, shuffle, borrow=True):
    data_x, data_y = data_xy
    
    if(shuffle):
        np.random.seed(42)
        idx = np.arange(data_x.shape[0])
        np.random.shuffle(idx)
        data_x = data_x[idx,:]
        data_y = data_y[idx]
    
    shared_x = theano.shared(np.asarray(data_x, dtype = 'float32'),
                                 borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y, dtype='int32'),
                                 borrow=borrow)
    return shared_x, shared_y
 
def load_data(dataset):    
    f = open(dataset, 'rb')
    train_set, valid_set = cPickle.load(f)
    f.close()
              
    train_set_x, train_set_y = shared_dataset(train_set, shuffle=True)
    valid_set_x, valid_set_y = shared_dataset(valid_set, shuffle=False)
       
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]
    return rval

def load_valid_sizes (dataset):
    f = open(dataset, 'rb')
    valid_sizes = cPickle.load(f)
    f.close()
    return valid_sizes
    
