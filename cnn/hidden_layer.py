import numpy
import theano
import theano.tensor as T


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, dropout_prob, W=None, b=None):
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
            lin_output = self._dropout(rng, lin_output, dropout_prob)

        self.output = T.tanh(lin_output)

        #print self.W.get_value()
        self.params = [self.W, self.b]

    def _dropout(self, rng, layer, p):
        srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
        mask = srng.binomial(n=1, p=1 - p, size=layer.shape)
        output = layer * T.cast(mask, 'float32')
        return output