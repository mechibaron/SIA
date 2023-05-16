import numpy as np

import utils
from constants import FIRST, MIDDLE, LAST
from kohonen import kohonen_alg
import matplotlib.pyplot as plt
import matplotlib


def ej1(learning_rate, epochs, type_model, similitud, radio, k):
    input_names, inputs, categories = utils.import_data('data/europe.csv')
    
    # test_idx = 3
    country_name_train = np.array(input_names)
    # country_name_train = np.delete(country_name_train,test_idx)

    
    training_set = np.array(inputs, dtype=float)
    # training_set = np.delete(training_set,test_idx, axis=0)

    p = len(training_set)
    n = len(training_set[0])
    if(type_model == 'kohonen'): 
        model = kohonen_alg.Kohonen(p, n, k, radio, learning_rate, similitud, epochs,training_set)
        neurons_weights, neurons_countries = model.train_kohonen()

        country_name_train = set(country_name_train.tolist())
        country_labels = list(set(country_name_train))

        fig, ax = plt.subplots(1, 1)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "orange", "red"])
        im = ax.imshow(neurons_weights, cmap=cmap)

        for j in range(k**2):
            winner_pos = np.array(np.unravel_index(j, neurons_weights.shape))
            country_label = ""
            for idx in range(p):
                if(neurons_countries[idx] == j):
                    country_label = country_label + country_labels[idx] + '\n'
            ax.text(winner_pos[1], winner_pos[0], country_label, ha="center", va="center", color="black", fontsize=5)

        fig.colorbar(im)
        plt.title(f'Grilla de neuronas de {k}x{k} ')
        ax.yaxis.set_major_locator(plt.NullLocator())  # remove y axis ticks
        ax.xaxis.set_major_locator(plt.NullLocator())  # remove x axis ticks
        plt.show()

        # Categories
        for categoryIdx in range(len(categories)):
            print(categoryIdx)
            # categoryIdx = 3
            train_category = [fila[categoryIdx] for fila in training_set]
            print(train_category)

            p = len(train_category)
            n = 1
            model = kohonen_alg.Kohonen(p, n, k, radio, learning_rate, similitud, epochs,train_category)
            neurons_weights, neurons_countries = model.train_kohonen_per_category()

            fig, ax = plt.subplots(1, 1)
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "orange", "red"])
            im = ax.imshow(neurons_weights, cmap=cmap)


            for j in range(k**2):
                winner_pos = np.array(np.unravel_index(j, neurons_weights.shape))
                country_label = ""
                for idx in range(p):
                    if(neurons_countries[idx] == j):
                        country_label = country_label + country_labels[idx] + ": " + str(train_category[idx]) + '\n'
                ax.text(winner_pos[1], winner_pos[0], country_label, ha="center", va="center", color="black", fontsize=5)

            fig.colorbar(im)
            plt.title(f'Grilla de neuronas de {k}x{k} para categoria: {categories[categoryIdx]}')
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