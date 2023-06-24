import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from multilayer.multilayer_perceptron import *
from multilayer.layer import Layer
from plotting import calculate_error

from res.fonts import *
from src.utils import *
import utils_1
import json

with open('./config.json', 'r') as f:
    data = json.load(f)
    f.close()

momentum, eta, epochs = utils_1.getDataFromFile(data)

x = np.array(get_input(2))
x = [x[6], x[11], x[19], x[9], x[10]]
x = np.array(x)

x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)

x = transform(x)



layer1 = Layer(20, 35, activation="tanh")
layer2 = Layer(10, activation="tanh")
layer3= Layer(5, activation="tanh")
layer4 = Layer(2, activation="tanh")
layer5 = Layer(5, activation="tanh")
layer6 = Layer(10, activation="tanh")
layer7 = Layer(20, activation="tanh")
layer8 = Layer(35, activation="tanh")

# layers = [layer1, layer4, layer7, layer8] #35-20-2-20-35
# layers = [layer1, layer2, layer4, layer6, layer7, layer8] #35-20-10-2-10-20-35
layers = [layer1, layer2, layer3, layer4,layer5, layer6, layer7, layer8]  #35-20-10-5-2-5-10-20-35

encoderDecoder = MultilayerPerceptron(layers, init_layers=True, momentum=momentum, eta=eta)

min_error, errors, epoch, training_accuracies, latent_layer = encoderDecoder.train(x, x, iterations_qty=epochs, adaptative_eta=False)

encoder = MultilayerPerceptron(encoderDecoder.neuron_layers[0:int(len(layers)/2)], init_layers=False)

decoder = MultilayerPerceptron(encoderDecoder.neuron_layers[int(len(layers)/2):], init_layers=False)

error = []
error2 = []

x_noise = []
x_noise2 = []
probability_noise = 0.2
probability_noise2 = 0.05
for i in range(len(x)):
    x_noise.append(create_noise(x[i],probability_noise))
    x_noise2.append(create_noise(x[i],probability_noise2))

x_noise = np.array(x_noise)
x_noise2 = np.array(x_noise2)
print(x_noise)

for i in range(len(x)):
    print(i)
    expected = x[i, :]    
    to_predict = x_noise[i,:]
    to_predict2 = x_noise2[i,:]
    encoded = encoder.predict(to_predict)
    encoded2 = encoder.predict(to_predict2)
    decoded = decoder.predict(encoded)
    decoded2 = decoder.predict(encoded2)
    graph_digits_noisy(expected, to_predict,decoded)
    graph_digits_noisy(expected, to_predict2,decoded2)
    error.append(calculate_error(decoded, expected))
    error2.append(calculate_error(decoded2, expected))

mean_error = [np.mean(values) for values in error]
mean_error2 = [np.mean(values) for values in error2]
print("Error medio promedio: ",np.sum(mean_error)/len(mean_error))
print("Error medio promedio: ",np.sum(mean_error2)/len(mean_error2))