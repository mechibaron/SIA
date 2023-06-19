import numpy as np
import random
from src.methods import *

class MultilayerPerceptron:
    def __init__(self, neuron_layers, eta=0.001, delta=0.05, init_layers=True, momentum=False):
        self.momentum_number = 0.8
        self.alpha = eta * 0.1
        self.beta = eta * 0.1
        self.eta = eta
        self.delta = delta
        self.neuron_layers = neuron_layers
        self.momentum = momentum
        self.k = 4
        if init_layers:
            self.init_layers()

    def init_layers(self):
        for i in range(len(self.neuron_layers)):
            if i != 0:
                self.neuron_layers[i].init_weights(inputs=self.neuron_layers[
                    i - 1].neurons_qty) 
            else:
                self.neuron_layers[i].init_weights() 
            self.neuron_layers[i].momentum = self.momentum
            self.neuron_layers[i].alpha = self.momentum_number

    def predict(self, a_input):
        res = a_input
        for i in range(len(self.neuron_layers)):
            res = self.neuron_layers[i].forward(res)
        return res

    def calculate_mean_square_error(self, training_set, expected_set):
        su = 0
        for i in range(len(training_set)):
            x = training_set[i]
            y = expected_set[i]

            predicted = self.predict(x)
            aux = np.linalg.norm(predicted - y, ord=2) ** 2
            su += aux
        return su / len(training_set)

    def back_propagate(self, predicted, x, y):
        delta = None
        for i in reversed(range(len(self.neuron_layers))):
            if i == 0:
                v = x
            else:
                v = self.neuron_layers[i - 1].v
            if i != len(self.neuron_layers) - 1:
                dif = np.matmul(np.transpose(self.neuron_layers[i + 1].weights[:, 1:]), np.transpose(delta))
                dif = np.transpose(dif)
                dif = np.array(dif)
            else:
                dif = y - predicted

            delta = self.neuron_layers[i].back_propagate(dif, v, self.eta)
        return delta

    def train(self, training_set, expected_set, error_epsilon=0, iterations_qty=10000, adaptative_eta=False):
        training_set = np.array(training_set)
        expected_set = np.array(expected_set)
        ii = 0
        shuffled_list = [a for a in range(0, len(training_set))]
        random.shuffle(shuffled_list)
        Error = 1
        min_error = float("inf")
        errors = []
        training_accuracies = []
        epochs = []
        eta_iteration = 0
        while ii < iterations_qty and Error > error_epsilon:
            j = 0
            training_correct_cases = 0
            while j < len(training_set):
                x = training_set[shuffled_list[j]]
                y = expected_set[shuffled_list[j]]

                predicted_value = self.predict(x)  # forward propagation

                error = self.back_propagate(predicted_value, x, y)
                aux_training = 0

                for i in range(len(error)):
                    if error[i] < self.delta:
                        aux_training += 1
                if aux_training == len(error):
                    training_correct_cases += 1

                j += 1
            training_accuracies.append(training_correct_cases / len(training_set))
            Error = self.calculate_mean_square_error(training_set, expected_set)

            if adaptative_eta and len(errors) > 1:
                if (Error - errors[-1]) < 0:
                    if eta_iteration <= 0:
                        eta_iteration -= 1
                    else:
                        eta_iteration = 0
                elif (Error - errors[-1]) > 0:
                    if eta_iteration >= 0:
                        eta_iteration += 1
                    else:
                        eta_iteration = 0

                if eta_iteration < -self.k:
                    if self.eta + self.alpha * self.eta < 1000:  # MAX
                        self.eta += self.alpha
                elif eta_iteration > self.k:
                    if self.eta - self.beta * self.eta > -1000:  # MIN
                        self.eta -= self.beta * self.eta

            if Error < min_error:
                min_error = Error
            errors.append(Error)

            epochs.append(ii)
            ii += 1

        return min_error, errors, epochs, training_accuracies