import numpy as np

import utils
from constants import FIRST, MIDDLE, LAST
from kohonen import kohonen_alg
import matplotlib

def ej1(learning_rate, epochs, type_model, similitud, radio, k):
    input_names, inputs, categories = utils.import_data('data/europe.csv')
    
    country_name_train = np.array(input_names)
    # country_name_train = np.delete(country_name_train,3)
    
    training_set = np.array(inputs, dtype=float)
    # training_set = np.delete(training_set,3, axis=0)

    p = len(training_set)
    n = len(training_set[0])
    iterations = epochs * n
    if(type_model == 'kohonen'):
        model = kohonen_alg.Kohonen(p, n, k, radio, learning_rate, similitud, iterations,training_set)
        neurons_activation = model.train_kohonen()
        distances = np.zeros(shape=(k, k))
        for i in range(len(k)):
            for j in range(len(k)):
                distances[i][j] = model.get_neighbours_distance(i, j, k)

        fig, ax = plt.subplots(1, 1)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "orange", "red"])
        im = ax.imshow(neurons_activation, cmap=cmap)
        fig.colorbar(im)
        plt.title(f'Grilla de neuronas de {k}x{k} ')
        ax.yaxis.set_major_locator(plt.NullLocator())  # remove y axis ticks
        ax.xaxis.set_major_locator(plt.NullLocator())  # remove x axis ticks
        plt.show()

        fig, ax = plt.subplots(1, 1)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "grey"])
        im = ax.imshow(distances, cmap=cmap)
        fig.colorbar(im)
        plt.title(f'Media de distancia euclidea entre pesos de neuronas vecinas')
        ax.yaxis.set_major_locator(plt.NullLocator())  # remove y axis ticks
        ax.xaxis.set_major_locator(plt.NullLocator())  # remove x axis ticks
        plt.show()

    # else:
    #  modelo de oja
     
     
    # model.test(training_set[3], country_name_train, categories)