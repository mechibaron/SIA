import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

class Kohonen:

    def __init__(self,p, n, k,radio, learning_rate, similitud, epochs, X,country_name_train, categories):
        self.p = p
        self.n = n
        self.k = k
        self.neurons = np.zeros((k,k))
        self.neurons_reshape = self.neurons.reshape(k**2)
        self.weights = []
        # Input Related Weight
        for _ in range(k**2):
            index = np.random.randint(0, p-1)
            if(n==1):
                x = self.standard_i(X)
            else:
                x = self.standard_i(X[index])
            self.weights.append(x) 
        # Uniform Distributed Weight
        # self.weights = np.random.rand(k**2,n)
        self.radio = [radio,radio]
        self.learning_rate = [learning_rate,learning_rate]
        self.similitud = similitud
        self.epochs = epochs
        self.X = X
        self.country_name_train = country_name_train
        self.categories = categories
    
    def predict(self, input_data):
        activations = np.zeros((self.k, self.k))
        for x in input_data:
            winner_index = self.winner(x)
            activations[winner_index] += 1
        return activations   
    
    def standard(self, X):
        X_standard = []
        for i in range(len(X)):
            X_standard.append(self.standard_i(X[i]))    
        return X_standard
    
    def standard_i(self,x):
        mean =[np.mean(x) for _ in range(len(x))]
        std = [np.std(x) for _ in range(len(x))]
        return (x - np.array(mean))/np.array(std) 

    def get_neighbours_weight_distance(self, winner_pos):
        distances=[]
        for i in range(self.k):
            for j in range(self.k):
                distance = self.get_neighbours_distance(np.array(winner_pos), [i,j])
                # Veo si son vecinos
                if(distance <= self.radio[1]):
                    distances.append(distance)
        return np.mean(distances)

    
    def get_neighbours_distance(self, winner_pos, neurons):
        #winner_pos = [x,y]
        #neurons = [a,b]
        return np.linalg.norm(winner_pos - neurons)

    
    def regla_de_kohonen(self, distances, x):
        for j in range(self.k**2):
            # Si soy vecino actulizo mis pesos
            if(j in distances):
                for p in range(self.n):
                    self.weights[j][p] += self.learning_rate[1] * (x[p]-self.weights[j][p])  

    def regla_de_kohonen_per_category(self, distances, x):
        for j in range(self.k**2):
            # Si soy vecino actulizo mis pesos
            if(j in distances):
                self.weights[j] += self.learning_rate[1] * (x-self.weights[j])  

    def euclidea(self, x):
        w=[]
        for j in range(self.k**2): #recorriendo filas
            w.append(np.linalg.norm(x - self.weights[j]))
        wk = min(w)
        return w.index(wk)
    
    def exponencial(self, x):
        w=[]
        for j in range(self.k**2): #recorriendo filas
            w.append(np.exp(-((np.linalg.norm(x - self.weights[j]))**2)))
        wk = min(w)
        return w.index(wk)
    
    def winner(self, x):
        if(self.similitud == "euclidea"):
            return self.euclidea(x)
        else:
            return self.exponencial(x)
    
    def activation(self,j, epoch):
        if(epoch == self.epochs - 1):
            self.neurons_reshape[j]+=1
            self.neurons = self.neurons_reshape.reshape(self.k,self.k)

        winner_pos = np.unravel_index(j, self.neurons.shape)
        distances = []
        # Obtengo la distancia entre neuronas
        for i in range(self.k):
            for j in range(self.k):
                distance = self.get_neighbours_distance(np.array(winner_pos), [i,j])
                # Veo si son vecinos
                if(distance <= self.radio[1]):
                    distances.append(np.ravel_multi_index((i, j), self.neurons.shape))
        return distances
    
    def plot_heatmap(self, similitud, neurons_countries):
        fig, ax = plt.subplots(1, 1)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "orange", "red"])
        im = ax.imshow(self.neurons, cmap=cmap)

        for j in range(self.k**2):
            winner_pos = np.array(np.unravel_index(j, self.neurons.shape))
            country_label = ""
            for idx in range(self.p):
                if(neurons_countries[idx] == j):
                    country_label = country_label + self.country_name_train[idx] + '\n'
            ax.text(winner_pos[1], winner_pos[0], country_label, ha="center", va="center", color="black", fontsize=5)

        fig.colorbar(im)
        plt.title(f'Grilla de neuronas de {self.k}x{self.k} con similitud {similitud}')
        ax.yaxis.set_major_locator(plt.NullLocator())  # remove y axis ticks
        ax.xaxis.set_major_locator(plt.NullLocator())  # remove x axis ticks
        plt.show()

    def plot_category(self, categoryIdx, neurons_countries):
        train_category = [fila[categoryIdx] for fila in self.X]
        fig, ax = plt.subplots(1, 1)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "yellow", "green", "blue"])
        avg_matriz = np.zeros((self.k,self.k))
        for j in range(self.k**2):
            winner_pos = np.array(np.unravel_index(j, self.neurons.shape))
            country_label = ""
            avg_j = []
            for idx in range(self.p):
                if(neurons_countries[idx] == j):
                    avg_j.append(train_category[idx])
                    country_label = country_label + self.country_name_train[idx]
            avg_matriz[winner_pos[1], winner_pos[0]] = np.mean(avg_j)
        im = ax.imshow(avg_matriz, cmap=cmap)
        fig.colorbar(im)
        plt.title(f'Grilla de neuronas de {self.k}x{self.k} para categoria: {self.categories[categoryIdx]}')
        ax.yaxis.set_major_locator(plt.NullLocator())  # remove y axis ticks
        ax.xaxis.set_major_locator(plt.NullLocator())  # remove x axis ticks
        plt.show()

    def plot_u_matrix(self):
        distances = np.zeros(shape=(self.k, self.k))
        for i in range(self.k):
            for j in range(self.k):
                distances[i][j] = self.get_neighbours_weight_distance([i, j])
        fig, ax = plt.subplots(1, 1)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "grey"])
        im = ax.imshow(distances, cmap=cmap)
        fig.colorbar(im)
        plt.title(f'Media de distancia euclidea entre pesos de neuronas vecinas')
        ax.yaxis.set_major_locator(plt.NullLocator())  # remove y axis ticks
        ax.xaxis.set_major_locator(plt.NullLocator())  # remove x axis ticks
        plt.show()

    def barplot_x(self,X_standard):
        X = self.X
        X_n = X_standard
        area_x=[fila[0] for fila in X]
        gdp_x=[fila[1] for fila in X]
        inf_x=[fila[2] for fila in X]
        life_x=[fila[3] for fila in X]
        mil_x=[fila[4] for fila in X]
        pop_x=[fila[5] for fila in X]
        unem_x=[fila[6] for fila in X]

        area_xn=[fila[0] for fila in X_n]
        gdp_xn=[fila[1] for fila in X_n]
        inf_xn=[fila[2] for fila in X_n]
        life_xn=[fila[3] for fila in X_n]
        mil_xn=[fila[4] for fila in X_n]
        pop_xn=[fila[5] for fila in X_n]
        unem_xn=[fila[6] for fila in X_n]

        dfx= {'Area': area_x, 'GDP': gdp_x,'Inflation':inf_x,'Life Expect':life_x, 'Military': mil_x, 'Population Growth': pop_x, 'Unemployment': unem_x}
        dfx_data=pd.DataFrame(data=dfx, index = None)
        dfxn= {'Area': area_xn, 'GDP': gdp_xn,'Inflation':inf_xn,'Life Expect':life_xn, 'Military': mil_xn, 'Population Growth': pop_xn, 'Unemployment': unem_xn}
        dfxn_data=pd.DataFrame(data=dfxn, index = None)

        plt.figure(figsize=(25,13))
        plt.xlabel('Features',fontsize=15) 
        plt.ylabel('Value',fontsize=15)
        plt.title(('Non-Standarized Inputs'))
        dfx_data.boxplot(column=['Area', 'GDP', 'Inflation','Life Expect','Military','Population Growth','Unemployment'])
        plt.show()

        plt.figure(figsize=(25,13))
        plt.xlabel('Features',fontsize=15) 
        plt.ylabel('Value',fontsize=15)
        plt.title(('Standarized Inputs'))
        dfxn_data.boxplot(column=['Area', 'GDP', 'Inflation','Life Expect','Military','Population Growth','Unemployment'])
        plt.show()

        return None

    def train_kohonen(self):
        X_standard = self.standard(self.X)
        # self.barplot_x(X_standard)
        neuron_activations = np.zeros((self.k**2, len(X_standard)))
        neuron_country = np.zeros(len(X_standard))
        for i in range(self.epochs):
            print(i)
            for j in range(len(X_standard)):
                # Seleccionar un registro de entrada X^p
                x = X_standard[j]
                # Encontrar la neurona ganadora
                winner_index = self.winner(x)
                distances = self.activation(winner_index, i)
                # Actualizar los pesos segun kohonen
                self.regla_de_kohonen(distances, x)
                neuron_activations[winner_index][j] = 1
                neuron_country[j] = winner_index
            # Ajuste de radio:
            ajuste = self.radio[0] * (1 - i/self.epochs)
            self.radio[1] = 1 if ajuste < 1 else ajuste
            # Ajuste de ETA:
            self.learning_rate[1] = self.learning_rate[0] * (1 - i/self.epochs)
            # Segunda opcion de ajuste de ETA mas abruta (para ppt probar la dif entre las dos)
            # decay_rate es un hiperparametro que recibiriamos.
            # self.learning_rate[1] = self.learning_rate[0] * np.exp(-decay_rate*i)

        self.neurons = self.neurons_reshape.reshape(self.k,self.k)
        return neuron_country

    def train_kohonen_per_category(self):
        X_standard = self.standard_i(self.X)
        neuron_activations = np.zeros((self.k**2, len(X_standard)))
        neuron_country = np.zeros(len(X_standard))
        for i in range(self.epochs):
            print(i)
            for j in range(len(X_standard)):
                # Seleccionar un registro de entrada X^p
                x = X_standard[j]
                # Encontrar la neurona ganadora
                winner_index = self.winner(x)
                distances = self.activation(winner_index, i)
                # Actualizar los pesos segun kohonen
                self.regla_de_kohonen_per_category(distances, x)
                neuron_activations[winner_index][j] = 1

                print(f"Registro {j+1} asignado a la neurona {winner_index}")
                neuron_country[j] = winner_index
            # Ajuste de radio:
            ajuste = self.radio[0] * (1 - i/self.epochs)
            self.radio[1] = 1 if ajuste < 1 else ajuste
            # Ajuste de ETA:
            self.learning_rate[1] = self.learning_rate[0] * (1 - i/self.epochs)

        self.neurons = self.neurons_reshape.reshape(self.k,self.k)
        return neuron_country
