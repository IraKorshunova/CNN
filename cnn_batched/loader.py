import cPickle
import numpy
import theano

def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype='float32'),
                                 borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y, dtype='int32'),
                                 borrow=borrow)
    return shared_x, shared_y
 
def load_data(dataset):    
    f = open(dataset, 'rb')
    train_set, valid_set = cPickle.load(f)
    f.close()
    
    train_x, train_y = train_set
    
    idx_0 = numpy.where(train_y == 0)
    x0 = train_x[idx_0] 
    train_set_x0 = theano.shared(numpy.asarray(x0, dtype='float32'), borrow=True)
    
    idx_1 = numpy.where(train_y == 1)
    x1 = train_x[idx_1]
    train_set_x1 = theano.shared(numpy.asarray(x1, dtype='float32'), borrow=True) 
    
    valid_set_x, valid_set_y = shared_dataset(valid_set)
       
    rval = [(train_set_x0, train_set_x1), (valid_set_x, valid_set_y)]
    return rval

def load_valid_sizes (dataset):
    f = open(dataset, 'rb')
    valid_sizes = cPickle.load(f)
    f.close()
    return valid_sizes
    
