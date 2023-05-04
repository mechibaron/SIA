import numpy as np
import matplotlib.pyplot as plt
from multilayer.layer import Layer
from constants import *
from multilayer.activation_functions import Activation


class MultilayerPerceptron:
    adaptive_rate = False
    error_limit = 0.001
    prev_layer_neurons = 0

    def __init__(self, training_set, expected_output, learning_rate, learning_rate_params=None,
                 batch_size=1, momentum=True):
        # Training set example: [[1, 1], [-1, 1], [1, -1]]
        self.training_set = training_set
        # Expected output example: [[0, 0], [0, 1], [1, 0]]
        self.expected_output = expected_output
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.layers = []
        self.error_min = None
        self.momentum = momentum
        if learning_rate_params:
            self.adaptive_rate = True
            self.learning_rate_inc = learning_rate_params[0]
            self.learning_rate_dec = learning_rate_params[1]
            self.learning_rate_k = learning_rate_params[2]

    def train(self, epochs):
        error = 1
        errors_among_epochs=[]
        prev_error = None
        self.error_min = float('inf')
        k = 0
        aux_batch = self.batch_size
        acc_epochs = []
        for i in range(epochs):
            aux_training_set = self.training_set
            aux_expected_output = self.expected_output
            while len(aux_training_set) > 0:
                i_x = np.random.randint(0, len(aux_training_set))
                training_set = aux_training_set[i_x]
                expected_output = aux_expected_output[i_x]
                aux_training_set = np.delete(aux_training_set, i_x, axis=0)
                aux_expected_output = np.delete(aux_expected_output, i_x, axis=0)
                self.propagate(training_set)
                self.backpropagation(expected_output)
                aux_batch -= 1
                self.update_weights(aux_batch)
                if aux_batch == 0:
                    aux_batch = self.batch_size

                aux_error = self.calculate_error(expected_output)
                error += aux_error

                if self.adaptive_rate and prev_error:
                    k = self.adapt_learning_rate(error - prev_error, k)
                prev_error = aux_error
            error *= 0.5
            if error < self.error_min:
                self.error_min = error
            errors_among_epochs.append(self.error_min)
                # print(error)
            # if error < self.error_limit:
                # print("Error " + str(error))
                # return
            acc_epochs.append(self.test_input(self.training_set))

        # print("ACCURACY: \n", acc_epochs)

        print("Errores: ", errors_among_epochs)
        plt.plot(list(range(0,epochs)), errors_among_epochs)
        plt.title("Error vs Epochs, Learning rate 0.01")
        # plt.xlim(0, 200)
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.show()

    def get_accuracy(real_output, expected_output, criteria = None):
        results = np.zeros(2)
        for i in range(len(np.array(real_output))):
            if criteria is not None:
                right_answer = np.abs(real_output[i] - expected_output[i]) < 0.1
                # right_answer = criteria(real_output[i], expected_output[i])
            else:
                right_answer = real_output[i] == expected_output[i]
            if right_answer:
                results[0] += 1
            else:
                results[1] += 1
        right_answers = results[0]
        wrong_answers = results[1]
        print("obtuve acc")
        return right_answers / (right_answers + wrong_answers)

    def propagate(self, training_set):
        m = len(self.layers)
        self.layers[0].set_activations(training_set)
        for i in range(1, m):
            prev_layer = self.layers[i-1]
            self.layers[i].propagate(prev_layer)

    def calculate_error(self, expected_output):
        m = len(self.layers)
        neurons = self.layers[m - 1].neurons
        aux_sum = 0
        for i in range(len(neurons)):
            aux_sum += (expected_output[i] - neurons[i].activation) ** 2
        return aux_sum
    
    def backpropagation(self, expected_output):
        m = len(self.layers)
        for i in range(m - 1, 0, -1):
            neurons = self.layers[i].neurons
            for j in range(len(neurons)):
                if i == m - 1:
                    neurons[j].sigma = Activation.tanh_dx(neurons[j].excitation) * \
                                       (expected_output[j] - neurons[j].activation)
                else:
                    upper_layer_neurons = self.layers[i + 1].neurons
                    aux_sum = 0
                    for neuron in upper_layer_neurons:
                        aux_sum += neuron.weights[j] * neuron.sigma
                    neurons[j].sigma = Activation.tanh_dx(neurons[j].excitation) * aux_sum

    def update_weights(self, batch_size):
        m = len(self.layers)
        for i in range(1, m):
            neurons = self.layers[i].neurons
            prev_neurons_activations = self.layers[i - 1].get_neurons_activation()
            for neuron in neurons:
                neuron.update_weights(self.learning_rate, prev_neurons_activations, self.momentum, batch_size)

    def add(self, neurons, layer):
        self.layers.append(Layer(neurons, self.prev_layer_neurons, layer))
        self.prev_layer_neurons = neurons

    def adapt_learning_rate(self, delta_error, k):
        if delta_error < 0:
            if k > 0:
                k = 0
            k -= 1
            if k == -self.learning_rate:
                self.learning_rate += self.learning_rate_inc
        elif delta_error > 0:
            if k < 0:
                k = 0
            k += 1
            if k == self.learning_rate:
                self.learning_rate -= self.learning_rate_dec * self.learning_rate
        else:
            k = 0
        return k

    def test_input(self, test_set):
        output = []
        for i in range(len(test_set)):
            self.propagate(test_set[i])
            output.append([neuron.activation for neuron in self.layers[len(self.layers) - 1].neurons])
        return output
