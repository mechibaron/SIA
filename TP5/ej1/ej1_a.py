import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from multilayer.multilayer_perceptron import *
from multilayer.layer import Layer

from res.fonts import *
from src.utils import *
import json
import utils_1

def calculate_error(to_predict, decoded):
    error = 0
    for i in range(len(to_predict)):
        error += (to_predict[i] - decoded[i])**2
    return error/len(to_predict)

with open('./config.json', 'r') as f:
    data = json.load(f)
    f.close()

momentum, eta , epochs = utils_1.getDataFromFile(data)
# momentum, eta, epochs = utils_1.getDataFromFile(data)

etas = [0.0005]
# etas = [0.05, 0.005, 0.0005, 0.00005]
etas_label = ["0.0005"]
# etas_label = ["0.05", "0.005", "0.0005", "0.00005"]

# error = {}

for j in range(len(etas)):
    # print("Running for eta: ", etas[j])
    # error[etas_label[j]] = []
    for k in range(1):
        x = np.array(get_input(1))
        text_names = get_header(1)

        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)

        x = transform(x)

        layer1 = Layer(20, 35, activation="tanh")
        layer2 = Layer(10, activation="tanh")
        layer3 = Layer(2, activation="tanh")
        layer4 = Layer(20, activation="tanh")
        layer5 = Layer(10, activation="tanh")
        layer6 = Layer(35, activation="tanh")

        layers = [layer1, layer2, layer3, layer4, layer5, layer6]

        encoderDecoder = MultilayerPerceptron(layers, init_layers=True, momentum=momentum, eta=eta)
        # encoderDecoder = MultilayerPerceptron(layers, init_layers=True, momentum=momentum, eta=etas[j])
        
        min_error, errors, epoch, training_accuracies, latent_layer = encoderDecoder.train(x, x, iterations_qty=epochs, adaptative_eta=False)
        # print(encoderDecoder.neuron_layers)
    
        # print(min_error)

        encoder = MultilayerPerceptron(encoderDecoder.neuron_layers[0:int(len(layers)/2)], init_layers=False)

        decoder = MultilayerPerceptron(encoderDecoder.neuron_layers[int(len(layers)/2):], init_layers=False)

        aux_1 = []
        aux_2 = []

        tirada_error = []
        # Generate new character
        # Lo que hay en espacio latente y lo que decodifica
        decoded = decoder.predict(latent_layer)
        graph_digit(decoded)
            
        for i in range(len(x)):
            to_predict = x[i, :]
            encoded = encoder.predict(to_predict)
            print("encoded")
            print(encoded)
            decoded = decoder.predict(encoded)
            # graph_digits(to_predict, decoded)
            # aux_1.append(encoded[0])
            # aux_2.append(encoded[1])
            tirada_error.append(calculate_error(to_predict, decoded))
        # error[etas_label[j]].append(np.mean(calculate_error(to_predict, decoded)))
# print(error)
    
    # plt.xlim([-1.1, 1.1])
    # plt.ylim([-1.1, 1.1])
    # for i, txt in enumerate(text_names):
    #     plt.annotate(txt, (aux_1[i], aux_2[i]))
    # plt.scatter(aux_1, aux_2)
    # plt.show()
