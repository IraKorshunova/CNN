import theano
import numpy as np
from numpy import linalg as LA
from hidden_layer import HiddenLayer
from logreg_layer import LogisticRegressionLayer
import theano.tensor as T
from loader import load_data, load_valid_sizes
from lenet_layer import LeNetConvPoolLayer
import cPickle

class  Trainer(object):
    
    def __init__(self, data_file='../data/more_data.pkl',
                 valid_sizes_file='../data/valid_batch_sizes.pkl'):
        
        datasets = load_data(data_file)
    
        self.train_set_x0, self.train_set_x1 = datasets[0]
        n_pos = self.train_set_x1.get_value(borrow=True).shape[0]
        n_neg = self.train_set_x0.get_value(borrow=True).shape[0]
        self.train_batch_size = 2 * n_pos 
        self.n_train_batches = n_neg / n_pos
        self.half_batch_size = n_pos
        
        y = np.concatenate((np.zeros(n_pos, dtype='int32'), np.ones(n_pos, dtype='int32')))
        self.train_set_y = theano.shared(y, borrow=True)
            
        self.valid_set_x, self.valid_set_y = datasets[1]   
        self.valid_size = self.valid_set_x.get_value(borrow=True).shape[0]
        self.valid_sizes = load_valid_sizes(valid_sizes_file)
        self.n_valid_batches = self.valid_sizes.shape[0] 
        
        print '======== dataset'
        print 'train neg:', n_neg 
        print 'train pos:', n_pos
        print 'n training batches:', self.n_train_batches
     
        v = self.valid_set_y.get_value(borrow=True)
        print 'valid:', v.shape
        print 'valid number of seizures:', sum(v) 
        print 'valid file sizes:', self.valid_sizes
        print 'seizures per valid file:'
        begin = 0;
        for i in xrange(self.n_valid_batches):
            print i, sum(v[begin: begin + self.valid_sizes[i]])
            begin = begin + self.valid_sizes[i]   

    def evaluate_lenet(self, learning_rate=0.1, n_epochs=7,
                    n_timesteps=1000,
                    dim=18,
                    nkerns=[6, 16, 120],
                    recept_width=143,
                    pool_width=2):
    
        recept_width = (n_timesteps + 1 + pool_width) / (1 + pool_width + pool_width ** 2)
        print '======== params'
        print 'nkerns: ', nkerns
        print 'receptive width: ', recept_width
        print 'pool_width: ', pool_width
            
        rng = np.random.RandomState(23455)
                    
        index = T.lscalar()
        x = T.matrix('x')
        y = T.ivector('y') 
        batch_size = theano.shared(self.train_batch_size) 
        
        # 18@1*1000
        layer0_input = x.reshape((batch_size, dim, 1, n_timesteps))
    
        # image dim@1*1000
        # c1: nkerns[0]@1*858 (1000-143 +1)
        # s2: nkerns[0]@1*429
        # filter_shape: nkerns[0] - number of fm in c1, dim -number of fm in image, receptive field size = 1*143
        layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
                                    image_shape=(None, dim, 1, n_timesteps),
                                    filter_shape=(nkerns[0], dim, 1, recept_width),
                                    poolsize=(1, pool_width))
        
        # c3: 16@1*287 (429-143+1)
        # s4  16@1*143
        image_width = (n_timesteps - recept_width + 1) / pool_width
        layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
                                    image_shape=(None, nkerns[0], 1, image_width),
                                    filter_shape=(nkerns[1], nkerns[0], 1, recept_width),
                                    poolsize=(1, pool_width))
        
        # s4:(batch_size, 16, 1, 143) -> flatten(2) -> (batch_size, 16*1*143)  
        layer2_input = layer1.output.flatten(2)
    
        if(((n_timesteps - recept_width + 1) / pool_width - recept_width + 1) / pool_width != recept_width):
            recept_width = recept_width + 1
    
        # c5: 120@1*1
        layer2 = HiddenLayer(rng, input=layer2_input,
                         n_in=nkerns[1] * 1 * recept_width,
                         n_out=nkerns[2],
                         activation=T.tanh)
    
        # f6/output 
        layer3 = LogisticRegressionLayer(input=layer2.output, n_in=nkerns[2], n_out=2)
    
        cost = layer3.negative_log_likelihood(y)
    
        params = layer3.params + layer2.params + layer1.params + layer0.params
        grads = T.grad(cost, params)
        
        [g_W3, g_b3, g_W2, g_b2, g_W1, g_b1, g_W0, g_b0] = grads 
                   
        updates = []
        for param_i, grad_i in zip(params, grads):
            updates.append((param_i, param_i - learning_rate * grad_i))
                    
        train_model = theano.function([index], [cost, g_W3, g_b3, g_W2, g_b2, g_W1, g_b1, g_W0, g_b0],
                                      updates=updates,
                                      givens={
                                            x: T.concatenate([self.train_set_x0[index * self.half_batch_size: (index + 1) * self.half_batch_size],
                                            self.train_set_x1], axis=0),
                                            y: self.train_set_y})
    

        ber = layer3.ber(y)
        tp, tn = layer3.tptn(y)
        fp, fn = layer3.fpfn(y)
        
        validate_model_batch = theano.function([index], [tp, tn , fp, fn],
                givens={ 
                x: self.valid_set_x[index: index + batch_size],
                y: self.valid_set_y[index: index + batch_size]})
        
        validate_model = theano.function([], [ber , tp, tn , fp , fn],
                givens={ 
                x: self.valid_set_x,
                y: self.valid_set_y})
        
    
        #------------------------------  TRAINING
        patience = 10000 
        patience_increase = 2  
        improvement_threshold = 0.995  
        validation_frequency = min(self.n_train_batches, patience / 2)
        best_valid_ber = 1
        best_iter = 0
        
        self.cost = np.array([])
        self.norm_grads = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        f = open('out.pkl', 'wb')
                   
        epoch = 0
        done_looping = False
    
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            print 'e', epoch          
            
            for minibatch_index in xrange(self.n_train_batches):
                
                iter = (epoch - 1) * self.n_train_batches + minibatch_index
    
                [cost, gW3, gb3, gW2, gb2, gW1, gb1, gW0, gb0] = train_model(minibatch_index) 
                
                #print cost
                self.cost = np.append(self.cost, cost)           
                
                norms = [LA.norm(gW3), LA.norm(gb3), LA.norm(gW2), LA.norm(gb2), LA.norm(gW1), LA.norm(gb1), LA.norm(gW0), LA.norm(gb0)]
                #print norms
                self.norm_grads = np.vstack([self.norm_grads, norms])
                        
                # ------------------------ VALIDATION
                if (iter + 1) % validation_frequency == 0:
                    # for separate validation files
                    begin_index = 0
                    for i in xrange(self.n_valid_batches):
                        batch_size.set_value(self.valid_sizes[i])
                        [tp, tn, fp , fn] = validate_model_batch(begin_index)
                        begin_index = begin_index + self.valid_sizes[i]
                        print '-----', i
                        print 'tp:', tp, 'tn:', tn , 'fp:' , fp, 'fn', fn 
                                           
                    # for entire validation set
                    batch_size.set_value(self.valid_size)
                    [ber, tp, tn, fp, fn] = validate_model()
                    print '*********************'
                    print 'tp:', tp, 'tn:', tn , 'fp:' , fp, 'fn', fn 
                    if(np.isnan(ber)):
                        ber = 1
                    print 'ber:', ber
                    print '*********************'
                    
                    batch_size.set_value(self.train_batch_size)
                  
                    if ber < best_valid_ber:
                        if ber < best_valid_ber * improvement_threshold:
                            patience = max(patience, iter * patience_increase)
                        best_valid_ber = ber
                        best_iter = iter
    
                if patience <= iter:
                    done_looping = True
                    break
            
        print('Optimization complete.')
        print 'Best BER', best_valid_ber
        print 'Best iteration', best_iter + 1
        
        out = self.cost, self.norm_grads
        cPickle.dump(out, f)
        f.close()
            
        return  best_valid_ber

if __name__ == '__main__':
    Trainer().evaluate_lenet()
 
