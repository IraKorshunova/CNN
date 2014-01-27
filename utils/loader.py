import cPickle
import numpy
import theano

def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype = 'float32'),
                                 borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y, dtype='int32'),
                                 borrow=borrow)
    return shared_x, shared_y
 
def load_data(dataset):    
    f = open(dataset, 'rb')
    train_set, valid_set = cPickle.load(f)
    f.close()
           
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
       
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]
    return rval

def load_valid_sizes (dataset):
    f = open(dataset, 'rb')
    valid_sizes = cPickle.load(f)
    f.close()
    return valid_sizes
