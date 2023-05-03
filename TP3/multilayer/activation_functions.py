import numpy as np


class Activation:

    @staticmethod
    def sigmoid(x):
        if -700 < x < 700:
            return np.exp(x) / (1 + np.exp(x))
        return 0 if x < 0 else 1

    @staticmethod
    def sigmoid_dx(x):
        # se hace 0 despues de este valor
        if -355 < x < 355:
            return np.exp(x) / np.power(np.exp(x) + 1, 2)
        return 0

    @staticmethod
    def tanh(excitation):
        return np.tanh(excitation)

    @staticmethod
    def tanh_dx(excitation):
        return 1 - np.tanh(excitation) ** 2
