import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from multilayer.multilayer_perceptron import *
from multilayer.layer import Layer
from src.plotting import calculate_error

from res.fonts import *
from src.utils import *
import json
import utils_1

with open('./config.json', 'r') as f:
    data = json.load(f)
    f.close()

momentum, eta, epochs = utils_1.getDataFromFile(data)

x = np.array(get_input(2))
text_names = get_header(2)

x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)

x = transform(x)

# layer0 = Layer(2, 35, activation="tanh")
layer1 = Layer(20, 35, activation="tanh")
layer2 = Layer(10, activation="tanh")
layer3 = Layer(2, activation="tanh")
layer4 = Layer(10, activation="tanh")
layer5 = Layer(20, activation="tanh")
layer6 = Layer(35, activation="tanh")

# layers = [layer0, layer6]
layers = [layer1, layer2, layer3, layer4, layer5, layer6]
# layers = [layer1, layer3, layer5, layer6]

encoderDecoder = MultilayerPerceptron(layers, init_layers=True, momentum=momentum, eta=eta)

min_error, errors, epochs, training_accuracies = encoderDecoder.train(x, x, iterations_qty=epochs, adaptative_eta=False)

encoder = MultilayerPerceptron(encoderDecoder.neuron_layers[0:int(len(layers)/2)], init_layers=False)

decoder = MultilayerPerceptron(encoderDecoder.neuron_layers[int(len(layers)/2):], init_layers=False)

aux_1 = []
aux_2 = []
error = []

# Get new caracter -> 3
# x = np.array(get_input(0))
# text_names = get_header(0)

# x_mean = np.mean(x, axis=0)
# x_std = np.std(x, axis=0)

# x = transform(x)

for i in range(len(x)):
    to_predict = x[i, :]
    encoded = encoder.predict(to_predict)
    decoded = decoder.predict(encoded)
    aux_1.append(encoded[0])
    aux_2.append(encoded[1])
    graph_digits(to_predict, decoded)
    error.append(calculate_error(to_predict, decoded))
mean_error = [np.mean(values) for values in error]
print("Error medio promedio: ",np.sum(mean_error)/len(mean_error))

# plt.xlim([-1.1, 1.1])
# plt.ylim([-1.1, 1.1])
for i, txt in enumerate(text_names):
    plt.annotate(txt, (aux_1[i], aux_2[i]))
plt.scatter(aux_1, aux_2)
plt.show()