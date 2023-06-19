import numpy as np
from src.methods import *

class Layer:
    def __init__(self, neurons_qty, inputs=None, activation="tanh"):
        self.neurons_qty = neurons_qty
        self.inputs = inputs
        (f, df) = self.get_functions(activation)
        self.f = f
        self.df = df
        self.weights = None
        self.h = None
        self.v = None
        self.momentum = False
        self.alpha = None
        self.last_dw = 0

    def init_weights(self, inputs=None):
        self.inputs = inputs if inputs is not None else self.inputs
        self.weights = 2 * np.random.random((self.neurons_qty, self.inputs + 1)) - 1

    def get_functions(self, activation_function):
        if activation_function == "tanh":
            f = tanh_act
            df = der_tanh_act
        elif activation_function == "sigmoid":
            f = sigmoid_act
            df = der_sigmoid_act
        elif activation_function == "linear":
            f = lineal_act
            df = der_lineal_act
        else:
            raise LookupError("falta funcion")
        return f, df

    def forward(self, a_input):
        a_input_biased = np.insert(a_input, 0, 1)
        output = np.matmul(self.weights, np.transpose(a_input_biased))  # h
        output = np.transpose(output)
        self.h = output
        output = self.f(output)
        self.v = output
        return output

    def back_propagate(self, dif, v, eta):
        v = np.insert(v, 0, 1)
        delta = np.multiply(self.df(self.h), dif)
        aux = v.reshape((-1, 1))
        d_w = eta * v.reshape((-1, 1)) * delta
        if self.momentum:
            self.weights = np.transpose(d_w) + self.weights + (self.alpha * np.transpose(self.last_dw))
        else:
            self.weights = self.weights + np.transpose(d_w)
        self.last_dw = d_w
        return delta