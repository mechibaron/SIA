import numpy as np
from multilayer.multilayer_perceptron import MultilayerPerceptron
import matplotlib.pyplot as plt
import font

class Autoencoder:
    def __init__(self,input_size, hidden_layers, training_set, expected_output,learning_rate, epochs, batch_size):
        self.input_size = input_size # matriz de nxd 31x7 
        # Encoder layers
        self.hidden_layers_encoder = hidden_layers # matriz de dimension de capas ocultas (ultima = latent size)
        # Latent size
        self.latent_size = hidden_layers.pop()
        #Decoder Layers
        self.hidden_layers_decoder = hidden_layers[::-1]
        self.hidden_layers_decoder.append(input_size) # matriz de dimension de capas ocultas (ultima = latent size)
        # Layers encoder + decoder
        self.hidden_layers = self.hidden_layers_encoder
        self.hidden_layers.extend(self.hidden_layers_decoder)
        # self.weights = np.array(input_size,self.latent_size) # matriz de kxd (asociada al decoder)
        self.training_set = training_set
        self.learning_rate = learning_rate
        self.expected_output = expected_output
        self.epochs = epochs
        self.batch_size = batch_size 


    def build(self):
        # perceptron = self.create_multilayer_perceptron_and_train(self.training_set, self.expected_output, self.learning_rate, self.epochs, self.hidden_layers, self.batch_size)        
        perceptron_encoder = self.create_multilayer_perceptron_and_train(self.training_set, self.expected_output, self.learning_rate, self.epochs, self.hidden_layers_encoder, self.batch_size)        
        encoded_neurons = []
        for i, neuron in enumerate(perceptron_encoder.layers[-1].neurons):
            # print(neuron)
            encoded_neurons.append(neuron.weights)
        # print(encoded_neurons)
        perceptron_decoder = self.create_multilayer_perceptron_and_train(encoded_neurons, self.training_set, self.learning_rate, self.epochs, self.hidden_layers_decoder, self.batch_size)        
        aux_1 = []
        aux_2 = []
        for i in range(len(self.training_set)):
            # print(i)
            to_predict = self.training_set[i, :]
            print("Predict: " , to_predict)
            encoded = perceptron_encoder.test(to_predict)
            print("Encoded: " , encoded)
            decoded = perceptron_decoder.test(encoded)
            print("Decoded: " , decoded)
            self.graph_digits(to_predict, decoded)
            aux_1.append(encoded[0])
            aux_2.append(encoded[1])
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
        for i, txt in enumerate(font.header):
            plt.annotate(txt, (aux_1[i], aux_2[i]))
        plt.scatter(aux_1, aux_2)
        # plt.show()
        # self.build_encoder()
        # self.build_dencoder()

    
    def create_multilayer_perceptron_and_train(self,training_set, expected_output, learning_rate, epochs, layers, batch_size, momentum=False, adaptive_params=None):
        perceptron = MultilayerPerceptron(training_set, expected_output, learning_rate, adaptive_params, batch_size, momentum)
        perceptron.add(len(training_set[0]), 0)
        for i in range(len(layers)):
            perceptron.add(layers[i], 1)
        perceptron.add(len(expected_output[0]), 2)
        perceptron.train(epochs)
        return perceptron
    
    def graph_digits(self, original, output):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_title('Expected Result')
        ax2.set_title('AutoEncoder result')

    # def get_weights_from_layers(neurons):
    #     weights = []
    #     for i in range(len(neurons)):
    #         weights.append(neurons[i].weights)
    #     return weights

    def test():
        #compara el self.decoder con el training set
        pass