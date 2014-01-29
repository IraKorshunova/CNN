import theano
import numpy as np
import theano.tensor as T
from cnn.hidden_layer import HiddenLayer
from cnn.logreg_layer import LogisticRegressionLayer
from cnn.lenet_layer import LeNetConvPoolLayer
from data_iterator import DataIterator


class Trainer(object):
    
    def __init__(self, datasets):
        
        self.train_set_iterator = DataIterator(datasets[0])
        self.n_batches = self.train_set_iterator.get_number_of_batches()

        self.valid_set_x, self.valid_set_y = datasets[1]   
        self.valid_size = self.valid_set_x.shape[0]

        self.test_set_x, self.test_set_y = datasets[2]
        self.test_size = self.test_set_x.shape[0]

        print '======== dataset'
        print 'valid:', self.valid_set_x.shape
        print 'valid number of seizures:', sum(self.valid_set_y)

    def evaluate_lenet(self, learning_rate=0.01, n_epochs=10,
                    n_timesteps=1000,
                    dim=18,
                    nkerns= [6, 16, 120],  #[25, 85, 155],
                    recept_width=[32, 32], #32
                    pool_width=[5,5]): #5

        print '======== params'
        print 'nkerns: ', nkerns
        print 'receptive width: ', recept_width
        print 'pool_width: ', pool_width
            
        rng = np.random.RandomState(23455)
                    
        i = T.lscalar()
        x = T.matrix('x')
        y = T.ivector('y') 
        batch_size = theano.shared(1)
        
        # 18@1*1000
        layer0_input = x.reshape((batch_size, dim, 1, n_timesteps))
    
        # image 18 @ 1*1000
        # c1: nkerns[0] @ 1* (1000 - recept_width[0] + 1)
        # s2: nkerns[0] @ 1 * c1 / pool_width[0]
        layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
                                    image_shape=(None, dim, 1, n_timesteps),
                                    filter_shape=(nkerns[0], dim, 1, recept_width[0]),
                                    poolsize=(1, pool_width[0]))
        
        # c3: nkerns[1] @ 1 * (s2 - recept_width[1] + 1)
        # s4  nkerns[1] @ 1 *  c3 / pool_width
        input_layer1_width = (n_timesteps - recept_width[0] + 1) / pool_width[0]
        layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
                                    image_shape=(None, nkerns[0], 1, input_layer1_width),
                                    filter_shape=(nkerns[1], nkerns[0], 1, recept_width[1]),
                                    poolsize=(1, pool_width[1]))
        
        # s4:(batch_size, nkerns[1], 1, s4) -> flatten(2) -> (batch_size, nkerns[1]* 1 * s4)
        layer2_input = layer1.output.flatten(2)

        input_layer2_size = (input_layer1_width - recept_width[1] + 1) / pool_width[1]
        # c5: 120@1*1
        layer2 = HiddenLayer(rng, input=layer2_input,
                         n_in=nkerns[1] * 1 * input_layer2_size,
                         n_out=nkerns[2],
                         activation=T.tanh)
    
        # f6/output 
        layer3 = LogisticRegressionLayer(input=layer2.output, n_in=nkerns[2], n_out=2)
            
        cost = layer3.negative_log_likelihood(y)
    
        params = layer3.params + layer2.params + layer1.params + layer0.params
        grads = T.grad(cost, params)
    
        updates = []
        for param_i, grad_i in zip(params, grads):
            updates.append((param_i, param_i - learning_rate * grad_i))

        train_model = theano.function([x, y], cost, updates=updates)

        valid_cost = layer3.weighted_negative_log_likelihood(y)
        tp, tn = layer3.tptn(y)
        fp, fn = layer3.fpfn(y)

        validate_model = theano.function([x, y], [valid_cost, tp, tn, fp, fn])
        test_model = theano.function([x, y], [tp, tn, fp, fn])
        
            
        #------------------------------  TRAINING
        patience = 2*self.n_batches + 1
        patience_increase = 2  
        improvement_threshold = 0.995  
        validation_frequency = self.n_batches
        best_valid_cost = np.inf
        best_iter = 0
        iter = 0
                   
        epoch = 0
        done_looping = False
    
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            print 'e', epoch
            for batch in self.train_set_iterator:
                iter += 1
                train_model(batch[0], batch[1])
                
                # ------------------------ VALIDATION
                if (iter + 1) % validation_frequency == 0:
                    batch_size.set_value(self.valid_size)
                    [valid_cost, tp, tn, fp, fn] = validate_model(self.valid_set_x, self.valid_set_y)
                    print 'validation'
                    print 'tp:', tp, 'tn:', tn , 'fp:' , fp, 'fn', fn
                    print 'cost:', valid_cost
                    print '*********************'

                    batch_size.set_value(self.test_size)
                    [tp, tn, fp, fn] = test_model(self.test_set_x, self.test_set_y)
                    print 'test'
                    print 'tp:', tp, 'tn:', tn , 'fp:' , fp, 'fn', fn
                    print '*********************'

                    batch_size.set_value(1)
                  
                    if valid_cost < best_valid_cost:
                        if valid_cost < best_valid_cost * improvement_threshold:
                            patience = max(patience, iter * patience_increase)
                        best_valid_cost = valid_cost
                        best_iter = iter
    
                if patience <= iter:
                    done_looping = True
                    break
            
        print('Optimization complete.')
        print 'Best validation cost', best_valid_cost
        print 'Best iteration', best_iter + 1
                        
        return np.array(best_valid_cost).item()

if __name__ == '__main__':
    Trainer().evaluate_lenet()
