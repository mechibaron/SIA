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
    
    training_set = np.array(inputs, dtype=float)
    training_set = np.delete(training_set,test_idx, axis=0)

    p = len(training_set)
    n = len(training_set[0])
    if(type_model == 'kohonen'): 
        model = kohonen_alg.Kohonen(p, n, k, radio, learning_rate, similitud, epochs,training_set)
        neurons_weights = model.train_kohonen()
        # for i in range(k):
        #     for j in range(k):
        #         distance = model.get_neighbours_distance(winner_pos, [i,j])

        fig, ax = plt.subplots(1, 1)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "orange", "red"])
        im = ax.imshow(neurons_weights, cmap=cmap)
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