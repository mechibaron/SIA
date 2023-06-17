import numpy as np
import matplotlib.pyplot as plt
from src.layer import Layer
from src.multilayer_perceptron import *
from res.fonts import *
from src.utils import *
import json
import utils_1

with open('./config.json', 'r') as f:
    data = json.load(f)
    f.close()

momentum, eta, epochs = utils_1.getDataFromFile(data)

x = np.array(get_input(1))
text_names = get_header(1)

x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)

x = transform(x)

layer1 = Layer(20, 35, activation="tanh")
#layer2 = Layer(20, activation="tanh")
layer3 = Layer(2, activation="tanh")
layer4 = Layer(20, activation="tanh")
#layer5 = Layer(30, activation="tanh")
layer6 = Layer(35, activation="tanh")

layers = [layer1, layer3, layer4, layer6]

encoderDecoder = MultilayerPerceptron(layers, init_layers=True, momentum=momentum, eta=eta)

min_error, errors, epochs, training_accuracies = encoderDecoder.train(x, x, iterations_qty=epochs, adaptative_eta=False)
print(min_error)

encoder = MultilayerPerceptron(encoderDecoder.neuron_layers[0:int(len(layers)/2)], init_layers=False)

decoder = MultilayerPerceptron(encoderDecoder.neuron_layers[int(len(layers)/2):], init_layers=False)

aux_1 = []
aux_2 = []

for i in range(len(x)):
    to_predict = x[i, :]
    encoded = encoder.predict(to_predict)
    decoded = decoder.predict(encoded)
    graph_digits(to_predict, decoded)
    aux_1.append(encoded[0])
    aux_2.append(encoded[1])

plt.xlim([-1.1, 1.1])
plt.ylim([-1.1, 1.1])
for i, txt in enumerate(text_names):
    plt.annotate(txt, (aux_1[i], aux_2[i]))
plt.scatter(aux_1, aux_2)
plt.show()