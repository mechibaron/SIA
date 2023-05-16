import numpy as np

import utils
from constants import FIRST, MIDDLE, LAST
from kohonen import kohonen_alg
import matplotlib.pyplot as plt
import matplotlib


def ej1(learning_rate, epochs, type_model, similitud, radio, k):
    input_names, inputs, categories = utils.import_data('data/europe.csv')
    
    test_idx = 3
    country_name_train = np.array(input_names)
    country_name_train = np.delete(country_name_train,test_idx)
    # country_name_train = country_name_train[:k**2]

    
    training_set = np.array(inputs, dtype=float)
    training_set = np.delete(training_set,test_idx, axis=0)

    p = len(training_set)
    n = len(training_set[0])
    if(type_model == 'kohonen'): 
        model = kohonen_alg.Kohonen(p, n, k, radio, learning_rate, similitud, epochs,training_set)
        neurons_weights = model.train_kohonen()
        # neurons_weights, neuron_activations = model.train_kohonen()

        country_name_train = set(country_name_train.tolist())

        # for i in range(k):
        #     for j in range(k):
        #         distance = model.get_neighbours_distance(winner_pos, [i,j])

        fig, ax = plt.subplots(1, 1)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "orange", "red"])
        im = ax.imshow(neurons_weights, cmap=cmap)
        country_labels = list(set(country_name_train))
        # num_labels = len(country_labels)
        # neuron_activations = model.predict(training_set)
        
        #este codigo imprime paises pero pierde paises, no imprime los 27
        country_labels = list(set(country_name_train))
        num_labels = len(country_labels)
        idx = 0
        for i in range(k):
            for j in range(k):
                if idx < num_labels:
                    country_label = country_labels[idx]
                else:
                    country_label = '-'
                ax.text(j, i, country_label, ha="center", va="center", color="black", fontsize=5)
                idx += 1
                
        #El siguiente codigo comentado es el que estaba intentando probar para que me tire mas de un pais dentro del cuadrado pero no funco
        # for i in range(k):
        #     for j in range(k):
        #         activated_countries = []
        #         for country_idx, activation in enumerate(neuron_activations[i * k + j]):
        #             if activation > 0:
        #                 activated_countries.append(country_labels[country_idx])
        #         country_label = '\n'.join(activated_countries) if activated_countries else '-'
        #         ax.text(j, i, country_label, ha="center", va="center", color="black", fontsize=5)

        fig.colorbar(im)
        plt.title(f'Grilla de neuronas de {k}x{k} ')
        ax.yaxis.set_major_locator(plt.NullLocator())  # remove y axis ticks
        ax.xaxis.set_major_locator(plt.NullLocator())  # remove x axis ticks
        plt.show()

        # fig, ax = plt.subplots(1, 1)
        # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "grey"])
        # im = ax.imshow(distances, cmap=cmap)
        # fig.colorbar(im)
        # plt.title(f'Media de distancia euclidea entre pesos de neuronas vecinas')
        # ax.yaxis.set_major_locator(plt.NullLocator())  # remove y axis ticks
        # ax.xaxis.set_major_locator(plt.NullLocator())  # remove x axis ticks
        # plt.show()

    # else:
    #  modelo de oja
     
     
    # model.test(training_set[3], country_name_train, categories)