import numpy


def sigmoid(inp):
    y = 1 + numpy.exp(-inp)
    z = 1 / y
    return z


def tanh(inp):
    y = 1 + numpy.exp(-2 * inp)
    z = 2 / y
    return z - 1


def softmax(inp):
    e_x = numpy.exp(inp - numpy.max(inp))
    return e_x / e_x.sum()


class ForwardLayer:
    def __init__(self, inp, n_n, activation):
        self.weights = numpy.random.randn(inp, n_n)
        self.activation = activation

    def forward(self, inp):
        x = inp.copy()
        out = numpy.dot(x, self.weights)
        out = self.activation(out)
        return out


class FeedForward:
    def __init__(self, input):
        self.inp = input
        self.layers = []

    def add_layer(self, n_n, act):
        if len(self.layers) == 0:

            layer = ForwardLayer(self.inp, n_n, activation=act)
            self.layers.append(layer)

        else:
            inp = self.layers[-1].weights.shape[1]
            layer = ForwardLayer(inp, n_n, act)
            self.layers.append(layer)

    def forward(self, inp):
        x = inp.copy()
        for l in self.layers:
            x = l.forward(x)

        return x

