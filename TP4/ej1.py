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
        print(neurons_countries)
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
        # X = df[features].values
        X = df.loc[:, features].values
        


        #inicio los pesos aleatoriamente
        n_features = X.shape[1]
        weights = np.random.randn(n_features)
        #normalizo los datos 
        X_standard = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        
        projection = np.dot(X_standard, weights)
        # Calculo la matriz de covarianza
        cov_matrix = np.cov(X_standard, rowvar=False)

        # Calculo los autovalores y autovectores
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Ordeno los autovectores en función de los autovalores
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Proyeccion de X_standard en autovectores
        projection = np.dot(X_standard, sorted_eigenvectors)

        # BARPLOT
        fig, ax = plt.subplots(figsize=(10, 6))
        ind = np.arange(len(countries))

        # Plotear las componentes principales para cada país
        positive_values = np.maximum(projection[:,0], 0)
        negative_values = np.minimum(projection[:,0], 0)
        # values = [positive_values[i] if positive_values[i] != 0 else negative_values[i] for i in range(len(positive_values))]
        # print("Component values: ",values)
        # print("Negative values: ",negative_values)
        ax.bar(ind, positive_values, color='b')
        ax.bar(ind, negative_values, color='r')
        ax.set_xlabel('País')
        ax.set_ylabel('Valor de la componente')
        ax.set_title('Componente principal para cada país')
        ax.set_xticks(ind)
        ax.set_xticklabels(countries, rotation=45)
        plt.tight_layout()
        plt.show()

        # Modelo de oja
        # components = []
        for e in range(epochs):
            for i in range(X_standard.shape[0]):
                x = X_standard[i]
                y = np.dot(x, weights)
                weights += learning_rate * y * (x - y * weights)
                #Normalizar los pesos despues de actualizarlos
                weights /= np.linalg.norm(weights)
            learning_rate = learning_rate / (e+1)
            # components.append(weights)

        component = weights
        sorted_indices = np.argsort(component)
        sorted_features = [features[i] for i in sorted_indices]

        #queria ver los valores no es necesario
        # print("Interpretación de la primera componente:")
        # for i in range(len(sorted_features)):
        #     print(f"{sorted_features[i]}: {first_component[i]: .4f}")

        # BIPLOT
        # Compute the second component
        # second_component = np.dot(X_standard, sorted_eigenvectors[:, 1])
        # # Compute the first component
        # first_component = np.dot(X_standard, sorted_eigenvectors[:, 0])

        # components_df = pd.DataFrame({'Component 1': first_component, 'Component 2': second_component, 'Country': countries})

        # # Plot the biplot
        # plt.figure(figsize=(10, 10))
        # num_points = len(components_df)
        # unique_colors = plt.cm.Set1(np.linspace(0, 1, num_points))
        # plt.scatter(components_df['Component 1'], components_df['Component 2'], marker='o',color=unique_colors)
        # plt.title("2 component OJA")
        # plt.xlabel("Principal Component 1", fontsize = 15)
        # plt.ylabel("Principal Component 2", fontsize = 15)
        # plt.grid()
        # # Add variable vectors (feature loadings) as arrows
        # for i, feature in enumerate(features):
        #     plt.arrow(0, 0, sorted_eigenvectors[i, 0], sorted_eigenvectors[i, 1], color='r', alpha=0.5)
        #     plt.text(sorted_eigenvectors[i, 0] * 1.15, sorted_eigenvectors[i, 1] * 1.15, feature, color = 'g', ha = 'center', va = 'center')

        # # Add country labels to the points
        # for i, country in enumerate(components_df['Country']):
        #     plt.annotate(country, (components_df['Component 1'][i], components_df['Component 2'][i]))    
        # plt.show()