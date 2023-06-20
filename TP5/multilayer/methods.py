import numpy as np

def sign_act(x):
    if x >= 0:
        return 1
    return -1

def der_sign_act(x):
    return 1

def lineal_act(x):
    return x

def der_lineal_act(x):
    return 1

def sigmoid_act(x):
    return 1 / (1 + np.exp(-1 * x))

def der_sigmoid_act(x):
    return sigmoid_act(x)*(1-sigmoid_act(x))

def tanh_act(x):
    return np.tanh(1 * x)

def der_tanh_act(x):
    return 1 / ((np.cosh(x)) ** 2)
