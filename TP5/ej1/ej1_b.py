import numpy as np
import matplotlib.pyplot as plt
from ..multilayer.layer import Layer
from ..multilayer.multilayer_perceptron import *
from res.fonts import *
from src.utils import *
import utils_1
import json

with open('./config.json', 'r') as f:
    data = json.load(f)
    f.close()

momentum, eta, epochs = utils_1.getDataFromFile(data)

x = np.array(get_input(1))
x = [x[6], x[11], x[19], x[9], x[10]]
x = np.array(x)

x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)

x_noise = noise(x)
x_noise2 = noise(x)
x = transform(x)

layer1 = Layer(20, 35, activation="tanh")
#layer2 = Layer(20, activation="tanh")
layer3 = Layer(2, activation="tanh")
layer4 = Layer(20, activation="tanh")
#layer5 = Layer(30, activation="tanh")
layer6 = Layer(35, activation="tanh")

layers = [layer1, layer3, layer4, layer6]

encoderDecoder = MultilayerPerceptron(layers, init_layers=True, momentum=momentum, eta=eta)

min_error, errors, epochs, training_accuracies = encoderDecoder.train(x_noise, x, iterations_qty=epochs, adaptative_eta=True)
print(min_error)

encoder = MultilayerPerceptron(encoderDecoder.neuron_layers[0:int(len(layers)/2)], init_layers=False)

decoder = MultilayerPerceptron(encoderDecoder.neuron_layers[int(len(layers)/2):], init_layers=False)

for i in range(len(x)):
    to_predict = x_noise2[i, :]
    encoded = encoder.predict(to_predict)
    decoded = decoder.predict(encoded)
    graph_digits(to_predict, decoded)