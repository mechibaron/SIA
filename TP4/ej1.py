import numpy as np

import utils
# from constants import FIRST, MIDDLE, LAST
from kohonen import kohonen_alg
import matplotlib.pyplot as plt
import pandas as pd
import json


def ej1(epochs):

    with open('./json/config_ej1.json', 'r') as f:
                ej1_data = json.load(f)
                f.close()
    learning_rate, type_model = utils.getDataFromEj1(ej1_data)
    
    if(type_model == 'kohonen'): 
        input_names, inputs, categories = utils.import_data('data/europe.csv')
        country_name_train = np.array(input_names)
        training_set = np.array(inputs, dtype=float)

        p = len(training_set)
        n = len(training_set[0])

        with open('./json/config_kohonen.json', 'r') as f:
            kohonen_data = json.load(f)
            f.close()
        similitud, radio, k = utils.getDataFromKohonen(kohonen_data)
        
        model = kohonen_alg.Kohonen(p, n, k, radio, learning_rate, similitud, epochs,training_set,country_name_train, categories)
        neurons_countries = model.train_kohonen()
        model.plot_heatmap(similitud, neurons_countries)

        # Categories Heatmap
        for categoryIdx in range(len(categories)):
            model.plot_category(categoryIdx, neurons_countries)

        # Matriz U
        model.plot_u_matrix(similitud)

        # Categories Train
        # for categoryIdx in range(len(categories)):
        #     train_category = [fila[categoryIdx] for fila in training_set]
        #     print(train_category)

        #     p = len(train_category)
        #     n = 1
        #     model = kohonen_alg.Kohonen(p, n, k, radio, learning_rate, similitud, epochs,train_category,country_name_train, categories)
        #     neurons_countries = model.train_kohonen_per_category()

        #     fig, ax = plt.subplots(1, 1)
        #     cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "orange", "red"])
        #     im = ax.imshow(model.neurons, cmap=cmap)


        #     for j in range(k**2):
        #         winner_pos = np.array(np.unravel_index(j, model.neurons.shape))
        #         country_label = ""
        #         for idx in range(p):
        #             if(neurons_countries[idx] == j):
        #                 country_label = country_label + country_name_train[idx] + ": " + str(train_category[idx]) + '\n'
        #         ax.text(winner_pos[1], winner_pos[0], country_label, ha="center", va="center", color="black", fontsize=5)

        #     fig.colorbar(im)
        #     plt.title(f'Grilla de neuronas de {k}x{k} para categoria: {categories[categoryIdx]} con Uniform Distributed Weights')
        #     ax.yaxis.set_major_locator(plt.NullLocator())  # remove y axis ticks
        #     ax.xaxis.set_major_locator(plt.NullLocator())  # remove x axis ticks
        #     plt.show()

    else:
        df = pd.read_csv("data/europe.csv", 
                 names=['Country', 'Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment'], skiprows=[0])

        features = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']
        countries = df['Country'].values

        countries = df['Country'].tolist()
        X = df[features].values

        #inicio los pesos aleatoriamente
        n_features = X.shape[1]
        weights = np.random.randn(n_features)
        #normalizo los datos 
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        components = np.dot(X, weights)
        # Calculo la matriz de covarianza
        cov_matrix = np.cov(X, rowvar=False)

        # Calculo los autovalores y autovectores
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Ordeno los autovectores en función de los autovalores
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Calculo las componentes principales
        components = np.dot(X, sorted_eigenvectors)


        #Modelo de oja
        for e in range(epochs):
            for i in range(X.shape[0]):
                x = X[i]
                y = np.dot(x, weights)
                weights += learning_rate * y * (x - y * weights)
                #Normalizar los pesos despues de actualizarlos
                weights /= np.linalg.norm(weights)
            learning_rate = learning_rate / (e+1)
                
        first_component = weights
        sorted_indices = np.argsort(first_component)
        sorted_features = [features[i] for i in sorted_indices]

        #queria ver los valores no es necesario
        print("Interpretación de la primera componente:")
        for i in range(len(sorted_features)):
            print(f"{sorted_features[i]}: {first_component[i]: .4f}")


        #grafico
        fig, ax = plt.subplots(figsize=(10, 6))
        ind = np.arange(len(countries))

        # Plotear las componentes principales para cada país
        positive_values = np.maximum(components[:,0], 0)
        negative_values = np.minimum(components [:,0], 0)
        ax.bar(ind, positive_values, color='b')
        ax.bar(ind, negative_values, color='r')
        ax.set_xlabel('País')
        ax.set_ylabel('Valor de la componente')
        ax.set_title('Componente principal para cada país')
        ax.set_xticks(ind)
        ax.set_xticklabels(countries, rotation=45)
        plt.tight_layout()
        plt.show()
     
    # model.test(training_set[3], country_name_train, categories)