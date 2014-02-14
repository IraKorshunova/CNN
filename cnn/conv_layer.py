import numpy
import theano
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from theano import tensor as T


class ConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize):
        assert image_shape[1] == filter_shape[1]

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) / numpy.prod(poolsize))
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))

        W_values = numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype='float32')
        self.W = theano.shared(value=W_values, name='W', borrow=True)

        b_values = numpy.zeros((filter_shape[0],), dtype='float32')
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        # new code
        self.filter_shape = [filter_shape[k] for k in [0, 2, 1, 3]]
        self.W_shuffled = self.W.dimshuffle(0, 2, 1, 3)
        self.W_shuffled = self.W_shuffled[:, :, ::-1, :]

        self.input_shape = [image_shape[k] for k in [0, 2, 1, 3]]
        self.input_shuffled = input.dimshuffle(0, 2, 1, 3)

        conv_out = conv.conv2d(self.input_shuffled, self.W_shuffled, filter_shape=self.filter_shape,
                               image_shape=self.input_shape)

        #print 'input_shape', self.input_shape
        #print 'filter_shape',self.filter_shape
        self.conv_out_shape = conv_out.shape
        self.input_shuffled_shape = self.input_shuffled.shape

        pooled_out = downsample.max_pool_2d(conv_out, poolsize, ignore_border=True)

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]