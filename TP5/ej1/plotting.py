import numpy as np
import matplotlib.pyplot as plt
import json
import utils_1
from src.layer import Layer
from src.multilayer_perceptron import *
from src.utils import *

def plotAverages(vae, data, labels):
    avg,_,_ = vae.encoder.predict(data)
    colormap = plt.cm.get_cmap('plasma')
    plt.figure(figsize=(12, 10))
    sc = plt.scatter(avg[:, 0], avg[:, 1], c=labels, cmap=colormap)
    plt.colorbar(sc)
    plt.show()


def plotLatent(vae, n=10, figsize=15, digit_size=28):
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = np.linspace(-1.0, 1.0, n)
    grid_y = np.linspace(-1.0, 1.0, n)[::-1]
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z = np.array([[xi, yi]])
            output = vae.decoder.predict(z)
            digit = output[0].reshape(digit_size, digit_size)
            figure[
            i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size,
            ] = digit
    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.imshow(figure, cmap="Greys_r")
    plt.show()

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
    y_plot = errors

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
    vary_hidden_layer()