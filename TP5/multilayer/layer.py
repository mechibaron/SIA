from multilayer.neuron import Neuron
import numpy as np


class Layer:

    def __init__(self, neurons, prev_layer_neurons, layer):
        self.neurons = [Neuron(prev_layer_neurons, 0, layer) for i in range(neurons)]

    def __str__(self):
        return self.neurons

    def set_activations(self, training_set):
        # Training set: [0, 1, -1]
        # print(len(self.neurons))
        for i in range(len(self.neurons)):
            self.neurons[i].activation = training_set[i]

    def get_neurons_activation(self):
        return np.array(list(map(lambda neuron: neuron.activation, self.neurons)))

    def propagate(self, prev_layer):
        prev_layer_activations = prev_layer.get_neurons_activation()
        for neuron in self.neurons:
            neuron.activate(prev_layer_activations)
