import numpy as np

class Kohonen:

    def __init__(self,p, n, k,radio, learning_rate, similitud, epochs, X):
        self.p = p
        self.n = n
        self.k = k
        self.neurons = np.zeros((k,k))
        self.neurons_reshape = self.neurons.reshape(k**2)
        self.weights = []
        for _ in range(k**2):
            index = np.random.uniform(0,1)
            # index = np.random.rand(0,1)
            # index = np.random.randint(0, p-1)
            # x = self.standard_i(X[index])
            # self.weights.append(x) 
        self.weights = np.random.rand(k**2,n)
        self.radio = [radio,radio]
        self.learning_rate = [learning_rate,learning_rate]
        self.similitud = similitud
        self.epochs = epochs
        self.X = X
    
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
        if(self.similitud == "euclides"):
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
    
    def neurons_weights(self):
        w = []
        for j in range(self.k**2):
            w.append(np.mean(self.weights[j]))
        return np.reshape(w,(self.k,self.k))


    def train_kohonen(self):
        X_standard = self.standard(self.X)
        # X_standard = X
        neuron_activations = np.zeros((self.k**2, len(X_standard)))
        # print(neuron_activations.shape)
        # print(neuron_activations)
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

                # print(f"Registro {j+1} asignado a la neurona {winner_index}")
            # Ajuste de radio:
            ajuste = self.radio[0] * (1 - i/self.epochs)
            self.radio[1] = 1 if ajuste < 1 else ajuste
            # Ajuste de ETA:
            self.learning_rate[1] = self.learning_rate[0] * (1 - i/self.epochs)
            # Segunda opcion de ajuste de ETA mas abruta (para ppt probar la dif entre las dos)
            # decay_rate es un hiperparametro que recibiriamos.
            # self.learning_rate[1] = self.learning_rate[0] * np.exp(-decay_rate*i)

        self.neurons = self.neurons_reshape.reshape(self.k,self.k)
        return self.neurons
        # return self.neurons_weights()
        # return self.neurons_weights(), self.predict(self.X)[1]
