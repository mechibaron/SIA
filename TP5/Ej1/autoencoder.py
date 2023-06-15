import numpy as np
from multilayer.multilayer_perceptron import MultilayerPerceptron

class Autoencoder:
    def __init__(self,input_size, hidden_layers, training_set, expected_output,learning_rate, epochs, batch_size):
        self.input_size = input_size # matriz de nxd 31x7 
        self.hidden_layers_enconder = hidden_layers # matriz de dimension de capas ocultas (ultima = latent size)
        self.latent_size = hidden_layers.pop()
        self.hidden_layers_decoder = hidden_layers[::-1].append(input_size) # matriz de dimension de capas ocultas (ultima = latent size)
        # self.weights = np.array(input_size,self.latent_size) # matriz de kxd (asociada al decoder)
        self.training_set = training_set
        self.learning_rate = learning_rate
        self.expected_output = expected_output
        self.epochs = epochs
        self.batch_size = batch_size

        self.encoder = None #multilayer? (misma cantidad de hidden layers)
        self.decoder = None #multilayer?

        self.build()

    def build(self):
        self.build_encoder()
        # self.build_dencoder()

    def build_encoder(self):
        # print(self.traisning_set.shape)
        perceptron = self.create_multilayer_perceptron_and_train(self.training_set, self.expected_output, self.learning_rate, self.epochs, self.hidden_layers_enconder, self.batch_size)
        self.encoder = perceptron.layers[-1].neurons # array de dimension de latent size de "letras" [[[7]],[],...]
        print("Encoder weights: ", self.encoder)
    
    def build_dencoder(self):
        perceptron = self.create_multilayer_perceptron_and_train(self.encoder, self.training_set, self.learning_rate, self.epochs, self.hidden_layers_decoder, self.batch_size)
        self.decoder = perceptron.weights
        print("Decoder weights: ", self.decoder)

    
    def create_multilayer_perceptron_and_train(self,training_set, expected_output, learning_rate, epochs, layers, batch_size, momentum=False, adaptive_params=None):
        # print(training_set)
        # print(training_set[0].shape)
        perceptron = MultilayerPerceptron(training_set, expected_output, learning_rate, adaptive_params, batch_size, momentum)
        print(len(training_set))
        perceptron.add(len(training_set[0]), 0)
        for i in range(len(layers)):
            perceptron.add(layers[i], 1)
        perceptron.add(len(expected_output[0]), 2)
        perceptron.train(epochs)
        return perceptron
    
    def test():
        #compara el self.decoder con el training set
        pass