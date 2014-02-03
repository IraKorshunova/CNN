import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse


class LogisticRegressionLayer(object):
    def __init__(self, input, n_in, n_out, training_mode, dropout_prob):
        self.W = theano.shared(value=np.zeros((n_in, n_out), dtype='float32'),
                               name='W', borrow=True)
        self.b = theano.shared(value=np.zeros((n_out,), dtype='float32'),
                               name='b', borrow=True)

        # inv_dropout_prob = np.float32(1-dropout_prob)
        # self.p_y_given_x = ifelse(T.eq(training_mode, 1), T.nnet.softmax(T.dot(input, self.W) + self.b),
        #                           T.nnet.softmax(T.dot(input, inv_dropout_prob*self.W) + self.b))

        if dropout_prob != 0:
            inv_dropout_prob = np.float32(1-dropout_prob)
            self.p_y_given_x = ifelse(T.eq(training_mode, 1), T.nnet.softmax(T.dot(input, self.W) + self.b),
                                      T.nnet.softmax(T.dot(input, inv_dropout_prob*self.W) + self.b))
        else:
            self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def weighted_negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.nonzero(y), 1]) - T.mean(T.log(self.p_y_given_x)[T.nonzero(y - 1), 0])


    def tptn(self, y):
        tp = T.and_(T.eq(y, 1), T.eq(self.y_pred, 1)).sum()
        tn = T.and_(T.eq(y, 0), T.eq(self.y_pred, 0)).sum()
        return [tp, tn]

    def fpfn(self, y):
        fp = T.and_(T.eq(y, 0), T.eq(self.y_pred, 1)).sum()
        fn = T.and_(T.eq(y, 1), T.eq(self.y_pred, 0)).sum()
        return [fp, fn]
