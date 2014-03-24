import numpy
import theano
import theano.tensor as T
from theano.ifelse import ifelse


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, training_mode, dropout_prob, activation_function, W=None, b=None):
        self.input = input

        if W is None:
            W_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype='float32')

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype='float32')
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b

        if dropout_prob != 0.0:
            lin_output = ifelse(T.eq(training_mode, 1), self._dropout(rng, lin_output, dropout_prob), lin_output)

        self.output = activation_function(lin_output)
        self.params = [self.W, self.b]

    def _dropout(self, rng, layer, p):
        srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
        mask = srng.binomial(n=1, p=1 - p, size=layer.shape)
        output = layer * T.cast(mask, 'float32')
        return output