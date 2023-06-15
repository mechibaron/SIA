import numpy as np
from multilayer.activation_functions import Activation

BIAS = 1
FIRST = 0
MIDDLE = 1
LAST = 2

class Neuron:
    excitation = 0
    activation = 0
    sigma = None
    delta = 0

    def __init__(self, prev_layer_neurons, activation, layer):
        self.activation = activation
        if layer > FIRST:
            self.weights = np.random.uniform(-1, 1, prev_layer_neurons)

    def excite(self, prev_layer_activations):
        print(self.weights)
        print(prev_layer_activations)
        self.excitation = np.inner(self.weights, prev_layer_activations) + BIAS
        return self.excitation

    def activate(self, prev_layer_activations):
        self.activation = Activation.tanh(self.excite(prev_layer_activations))
        return self.activation

    def update_weights(self, learning_rate, prev_layer_activations, momentum, batch_size):
        delta_weights = (learning_rate * self.sigma) * prev_layer_activations
        if batch_size > 0:
            self.delta += delta_weights
        else:
            if momentum:
                delta_weights += 0.8 * self.delta
            self.weights += delta_weights
            self.delta = delta_weights
