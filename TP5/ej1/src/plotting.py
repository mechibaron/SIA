import numpy as np
import matplotlib.pyplot as plt
import json
import utils_1
import sys
sys.path.append("..")
from multilayer.multilayer_perceptron import *
from multilayer.layer import Layer
from src.utils import *

# plotea para distinta cantidad de epocas el promedio del error en epocas (por ej para 10 epocas tengo un array de errores por epoca -> un promedio de esos valores)
def plot_variation_epochs():

    epochs_variation = [10,100,1000,10000]
    epochs_variation_labels = ["10","100","1000","10000"]

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

    error = {}

    for i in range(len(epochs_variation)):

        error[epochs_variation[i]] = []

        encoderDecoder = MultilayerPerceptron(layers, init_layers=True, momentum=momentum, eta=eta)

        min_error, errors, epochs, training_accuracies = encoderDecoder.train(x, x, iterations_qty=epochs_variation[i], adaptative_eta=False)

        encoder = MultilayerPerceptron(encoderDecoder.neuron_layers[0:int(len(layers)/2)], init_layers=False)

        decoder = MultilayerPerceptron(encoderDecoder.neuron_layers[int(len(layers)/2):], init_layers=False)

        # print(error)

        for j in range(len(x)):
            to_predict = x[j, :]
            # print("Predict: \n",to_predict)
            encoded = encoder.predict(to_predict)
            decoded = decoder.predict(encoded)
            # print("Decoded: \n",decoded)
            # print(i, calculate_error(to_predict, decoded))
            error[epochs_variation[i]].append(calculate_error(to_predict, decoded))
    print(error)
    x_plot = epochs_variation_labels
    y_plot = [np.mean(values) for values in error.values()]
    # for key, values in error.items():
    #     x_plot.extend([key] * len(values))
    #     y_plot.extend(np.mean(values))

    # Plot scatter
    plt.scatter(x_plot, y_plot)

    # Customize the plot
    plt.title("Mean error for different epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Mean error")

    # Show the plot
    plt.show()

#plote el error por epocas para 10000 epocas 
def plot_error_by_epochs():

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

    error = []

    encoderDecoder = MultilayerPerceptron(layers, init_layers=True, momentum=momentum, eta=eta)

    min_error, errors, epochs, training_accuracies = encoderDecoder.train(x, x, iterations_qty=10000, adaptative_eta=False)

    encoder = MultilayerPerceptron(encoderDecoder.neuron_layers[0:int(len(layers)/2)], init_layers=False)

    decoder = MultilayerPerceptron(encoderDecoder.neuron_layers[int(len(layers)/2):], init_layers=False)

    # print(error)

    for j in range(len(x)):
        to_predict = x[j, :]
        # print("Predict: \n",to_predict)
        encoded = encoder.predict(to_predict)
        decoded = decoder.predict(encoded)
        # print("Decoded: \n",decoded)
        # print(i, calculate_error(to_predict, decoded))
        error.append(calculate_error(to_predict, decoded))

    print(errors)
    x_plot = [i for i in range(10000)]
    # y_plot = errors #error
    y_plot = [np.mean(values) for values in error.values()] #mean error


    # Plot scatter
    plt.scatter(x_plot, y_plot)

    # Customize the plot
    plt.title("Error by epochs for 10000 epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Error")

    # Show the plot
    plt.show()

def calculate_error(to_predict, decoded):
    error = 0
    for i in range(len(to_predict)):
        error += (to_predict[i] - decoded[i])**2
    return error/len(to_predict)

def vary_hidden_layer():
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
    layer2 = Layer(10, activation="tanh")
    layer3 = Layer(2, activation="tanh")
    layer4 = Layer(10, activation="tanh")
    layer5 = Layer(20, activation="tanh")
    layer6 = Layer(35, activation="tanh")

    layers = [layer1,layer2, layer3, layer4, layer5,layer6]

    error = []

    encoderDecoder = MultilayerPerceptron(layers, init_layers=True, momentum=momentum, eta=eta)

    min_error, errors, epochs, training_accuracies = encoderDecoder.train(x, x, iterations_qty=10000, adaptative_eta=False)

    encoder = MultilayerPerceptron(encoderDecoder.neuron_layers[0:int(len(layers)/2)], init_layers=False)

    decoder = MultilayerPerceptron(encoderDecoder.neuron_layers[int(len(layers)/2):], init_layers=False)

    aux_1 = []
    aux_2 = []

    for j in range(len(x)):
        to_predict = x[j, :]
        # print("Predict: \n",to_predict)
        encoded = encoder.predict(to_predict)
        decoded = decoder.predict(encoded)
        # print("Decoded: \n",decoded)
        # print(i, calculate_error(to_predict, decoded))
        graph_digits(to_predict, decoded)
        print(j)
        aux_1.append(encoded[0])
        aux_2.append(encoded[1])

    # plt.xlim([-1.1, 1.1])
    # plt.ylim([-1.1, 1.1])
    # for i, txt in enumerate(text_names):
    #     plt.annotate(txt, (aux_1[i], aux_2[i]))
    # plt.scatter(aux_1, aux_2)
    plt.show()

if __name__ == '__main__':
    plot_error_by_epochs()