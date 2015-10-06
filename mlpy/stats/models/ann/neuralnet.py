from __future__ import division, print_function, absolute_import
# noinspection PyUnresolvedReferences
from six.moves import range

import numpy as np


# noinspection PyPep8Naming
class tanh(object):
    def __call__(self, x):
        return np.tanh(x)

    # noinspection PyMethodMayBeStatic
    def derivative(self, x):
        return 1.0 - x ** 2


# noinspection PyPep8Naming
class sigmoid(object):
    def __call__(self, x):
        return 1. / (1 + np.exp(-x))

    def derivative(self, x):
        sigma = self(x)
        return sigma * (1 - sigma)


# noinspection PyPep8Naming
class linear(object):
    def __call__(self, x):
        return np.copy(x)

    # noinspection PyMethodMayBeStatic
    def derivative(self, x):
        x = np.asarray(x)
        if x.size == 1:
            x.shape = (1,)
        return np.ones(x.shape)


# noinspection PyPep8Naming
class threshold(object):
    def __call__(self, x):
        x_ = np.asarray(x)
        if x_.size == 1:
            x_.shape = (1,)
        np.place(x_, x_ > 0, [1])
        np.place(x_, x_ < 0, [-1])
        return x_

    # noinspection PyMethodMayBeStatic
    def derivative(self, x):
        x = np.asarray(x)
        if x.size == 1:
            x.shape = (1,)
        return np.ones(x.shape)


class NeuralNetwork(object):
    """Neural network

    Parameters
    ----------
    layer_size : array_like, shape(`nlayers`,)
        Number of nodes per layer, where `nlayers` is equal to `nhidden` + 2.
    activation : array_like, shape(`nlayers`,)
        The function code specifying the function to use per layer.

    """

    def __init__(self, layers, activation=None):
        num_layers = len(layers)

        if activation is None:
            activation = ["linear"]
            for i in range(1, num_layers - 1):
                activation.append("tanh")
            activation.append("linear")

        activation = activation if isinstance(activation, list) else [activation]
        if not all(x in ["tanh", "sigmoid", "linear"] for x in activation):
            raise ValueError("Unknown layer activation function type")

        self._in = [np.empty((1,))] * num_layers
        self._out = [np.empty((1,))] * num_layers
        self._weights = []
        self._activation = []
        self._layers = layers

        for i in range(num_layers):
            if i < num_layers - 1:
                size = (layers[i] + 1, layers[i + 1])
                # if i < num_layers - 2:
                #     # add weight for bias unit
                #     size = (layers[i] + 1, layers[i + 1] + 1)
                self._weights.append(self.randomize_weights(-.3, .3, size))

            switch = {
                'tanh': tanh,
                'sigmoid': sigmoid,
                'linear': linear
            }
            self._activation.append(switch[activation[i]]())

        self._dirty = False

    def __str__(self):
        out = ""
        for i in range(len(self._weights)):
            out += self._print_layer(i)
        return out

    # def __repr__(self):
    #     return ""

    def get_activation(self, layer, node):
        return np.asarray(self._out[layer])[node]

    def get_activations(self, layer):
        return np.asarray(self._out[layer])

    # noinspection PyMethodMayBeStatic
    def randomize_weights(self, min_, max_, size):
        return min_ + (max_ - min_) * np.random.random(size)

    def feed_forward(self, input_):
        input_ = np.asarray(input_)
        if input_.size == 1:
            input_.shape = (1,)

        if input_.ndim != 1 or input_.shape[0] != self._layers[0]:
            raise ValueError("Array `input_` must be a vector of length %d." % self._layers[0])

        input_ = self._add_bias_unit(input_)
        self._in[0] = input_
        self._out[0] = self._activation[0](input_)

        for i in range(len(self._weights)):
            if i < len(self._weights) - 1:
                self._in[i + 1] = self._add_bias_unit(np.dot(self._out[i], self._weights[i]))
                self._out[i + 1] = self._add_bias_unit(self._activation[i + 1](self._in[i + 1][:-1]))
            else:
                self._in[i + 1] = np.dot(self._out[i], self._weights[i])
                self._out[i + 1] = self._activation[i + 1](self._in[i + 1])

        self._dirty = True
        return self._out[-1]

    def back_propagate(self, input_, target, learning_rate=0.01):
        input_ = np.asarray(input_)
        if input_.size == 1:
            input_.shape = (1,)

        if input_.ndim != 1 or input_.shape[0] != self._layers[0]:
            raise ValueError("Array `input_` must be a vector of length %d." % self._layers[0])

        target = np.asarray(target)
        if target.size == 1:
            target.shape = (1,)

        if target.ndim != 1 or target.shape[0] != self._layers[-1]:
            raise ValueError("Array `target` must be a vector of length %d." % self._layers[-1])

        if not self._dirty:
            self.feed_forward(input_)
        else:
            input_ = self._add_bias_unit(input_)
            if not np.any(input_ == self._in[0]):
                self.feed_forward(input_)

        error = target - self._out[-1]
        # self._back_propagate_error(error, learning_rate)

        error *= learning_rate
        error *= self._activation[-1].derivative(self._out[-1])
        for l in range(len(self._out) - 2, -1, -1):
            error = self._back_propagate_layer(l, error)

    def back_propagate_error(self, input_, error, learning_rate=0.01):
        input_ = np.asarray(input_)
        if input_.size == 1:
            input_.shape = (1,)

        if input_.ndim != 1 or input_.shape[0] != self._layers[0]:
            raise ValueError("Array `input_` must be a vector of length %d." % self._layers[0])

        error = np.asarray(error)
        if error.size == 1:
            error.shape = (1,)

        if error.ndim != 1 or error.shape[0] != self._layers[-1]:
            raise ValueError("Array `input_` must be a vector of length %d." % self._layers[-1])

        if not self._dirty:
            self.feed_forward(input_)
        else:
            input_ = self._add_bias_unit(input_)
            if not np.any(input_ == self._in[0]):
                self.feed_forward(input_)

        self._back_propagate_error(error, learning_rate)

    def _back_propagate_error(self, error, learning_rate):
        deltas = [error * self._activation[-1].derivative(self._out[-1])]

        for i in range(len(self._out) - 2, -1, -1):
            deltas.append(deltas[-1].dot(self._weights[i].T) * self._activation[i].derivative(self._out[i]))
        deltas.reverse()

        for i in range(len(self._weights)):
            layer = np.atleast_2d(self._out[i])
            delta = np.atleast_2d(deltas[i])
            self._weights[i] += learning_rate * layer.T.dot(delta)

        self._dirty = False

    def _back_propagate_layer(self, layer, error):
        ierror = error.dot(self._weights[layer][:-1].T) * self._activation[layer].derivative(self._out[layer][:-1])
        self._weights[layer] += np.tile(np.reshape(self._out[layer], (self._out[layer].shape[0], -1)),
                                        error.shape[0]) * error

        self._dirty = False
        return ierror

    # noinspection PyMethodMayBeStatic
    def _add_bias_unit(self, input_):
        in_ = np.ones((input_.shape[0] + 1,))
        in_[:-1] = input_
        return in_

    def _print_layer(self, layer):
        out = "layer %d" % layer

        out += " p:%d" % self._layers[layer]
        out += " n:%d\n" % self._layers[layer]

        for _ in range(len(self._weights)):
            if self._weights[layer].shape[0] > 1 and self._weights[layer].shape[1] > 1:
                out += str(self._weights[layer][:-1, :-1]) + '\n'
            elif self._weights[layer].shape[1] > 1:
                out += str(self._weights[layer][:-1]) + '\n'
            else:
                out += str(self._weights[layer][:-1, 0]) + '\n'

        if self._weights[layer].shape[0] > 1 and self._weights[layer].shape[1] > 1:
            out += 'B:' + str(self._weights[layer][2, :-1]) + '\n'
        elif self._weights[layer].shape[1] > 1:
            out += 'B:' + str(self._weights[layer][-1]) + '\n'
        else:
            out += 'B:' + str(self._weights[layer][-1, 0]) + '\n'

        return out
