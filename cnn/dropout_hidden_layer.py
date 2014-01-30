import theano.tensor as T
from cnn import hidden_layer


class DropoutHiddenLayer(hidden_layer.HiddenLayer):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        super(DropoutHiddenLayer, self).__init__(
            rng=rng, input=input, n_in=n_in, n_out=n_out,
            W=W, b=b, activation=activation)

        self.output = self._dropout(rng, self.output, p=0.5)

    def _dropout(self, rng, layer, p):
        srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
        mask = srng.binomial(n=1, p=1-p, size=layer.shape)
        output = layer * T.cast(mask, 'float32')
        return output