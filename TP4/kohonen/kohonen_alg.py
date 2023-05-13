import numpy as np

class Kohonen:

    def __init__(self,p, n, k,radio, learning_rate, similitud, iterations, X):
        self.p = p
        self.n = n
        # nostras pensamos => k = n => TODO revisar
        self.k = k
        self.neurons = np.zeros(k,k)
        self.neurons_reshape = self.neurons.reshape(k**2)
        self.weights = np.random.rand(k**2,n)
        self.radio = [radio,radio]
        self.learning_rate = [learning_rate,learning_rate]
        self.similitud = similitud
        self.iterations = iterations
        self.X = X
    

    # def euclidea (self, x,p):
    #     w=[]
    #     for j in range(self.k): #recorriendo filas
    #         w.append(np.abs(x - self.weights[j][p]))
    #     wk = min(w)
    #     return w.index(wk)
    
    # def exponencial (self, x, p):
    #     print(self.weights)
    #     w=[]
    #     for j in range(self.k): #recorriendo filas
    #         w.append(np.exp(-((x - self.weights[j][p])**2)))
    #     wk = min(w)
    #     return w.index(wk)
    
    # def regla_de_kohonen(self, winner_index,x, p):
    #     # print("REGLA DE KOHENEN")
    #     for j in range(self.k**2):
    #         if(np.abs(j-winner_index) <= self.radio[1]):
    #             # Si soy vecino actulizo mis pesos
    #             self.weights[j][p] += self.learning_rate[1] * (x-self.weights[j][p])  
        
    def standard(self, X):
        X_standard = []
        for i in range(len(X)):
            X_standard.append(self.standard_i(X[i]))    
        return X_standard
    
    def standard_i(self,x):
        mean =[np.mean(x) for _ in range(len(x))]
        std = [np.std(x) for _ in range(len(x))]
        return (x - np.array(mean))/np.array(std)    
    
    # def winner(self, x,p):
    #     if(self.similitud == "euclides"):
    #         return self.euclidea(x,p)
    #     else:
    #         return self.exponencial(x,p)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def get_distance(neuron1, neuron2):
        total_sum = 0
        for i in range(len(neuron1)):
            total_sum += (neuron1[i] - neuron2[i]) ** 2
        return np.sqrt(total_sum)
    
    def get_neighbours_distance(self, x, y, neurons):
        curr_neuron_weight = self.neurons[x][y]
        dist = 0
        size = 0
        if y + 1 < len(neurons):
            dist += self.get_distance(curr_neuron_weight, neurons[x][y + 1].weights)
            size += 1
        if y > 0:
            size += 1
            dist += self.get_distance(curr_neuron_weight, neurons[x][y - 1].weights)
        if x > 0:
            size += 1
            dist += self.get_distance(curr_neuron_weight, neurons[x - 1][y].weights)
        if x + 1 < len(neurons):
            size += 1
            dist += self.get_distance(curr_neuron_weight, neurons[x + 1][y].weights)
        return dist / size

    
    def regla_de_kohonen_2(self, winner_index,x):
        for j in range(self.k**2):
            # Si soy vecino actulizo mis pesos
            if(np.abs(j-winner_index) <= self.radio[1]):
                for p in range(self.n):
                    self.weights[j][p] += self.learning_rate[1] * (x-self.weights[j][p])  

    def euclidea_2 (self, x):
        w=[]
        for j in range(self.k**2): #recorriendo filas
            w.append(np.abs(x - self.weights[j]))
        wk = min(w)
        return w.index(wk)
    
    def exponencial_2 (self, x):
        w=[]
        for j in range(self.k**2): #recorriendo filas
            w.append(np.exp(-((x - self.weights[j])**2)))
        wk = min(w)
        return w.index(wk)
    
    def winner_2(self, x):
        if(self.similitud == "euclides"):
            return self.euclidea_2(x)
        else:
            return self.exponencial_2(x)
    
    def activation(self,j):
        self.neurons_reshape[j]+=1
        
    def train_kohonen(self):
        X_standard = self.standard(self.X)
        # X_standard = X
        for i in range(self.iterations):
            for j in range(len(X_standard)):
                # Seleccionar un registro de entrada X^p
                x = X_standard[j]
                # Encontrar la neurona ganadora
                winner_index = self.winner_2(x)
                self.activation(winner_index)
                # Actualizar los pesos segun kohonen
                self.regla_de_kohonen_2(winner_index, x)
                # for p in range(len(X_standard[j])):
                #     # Seleccionar un registro de entrada X^p
                #     x = X_standard[j][p]
                #     # Encontrar la neurona ganadora
                #     winner_index = self.winner(x,p)
                #     # Actualizar los pesos segun kohonen
                #     self.regla_de_kohonen(winner_index, x, p)
                
            # Ajuste de radio:
            ajuste = self.radio[0] * (1 - i/self.iterations)
            self.radio[1] = 1 if ajuste < 1 else ajuste
            # Ajuste de ETA:
            self.learning_rate[1] = self.learning_rate[0] * (1 - i/self.iterations)
            # Segunda opcion de ajuste de ETA mas abruta (para ppt probar la dif entre las dos)
            # decay_rate es un hiperparametro que recibiriamos.
            # self.learning_rate[1] = self.learning_rate[0] * np.exp(-decay_rate*i)

        self.neurons = self.neurons_reshape.reshape(self.k,self.k)
        return self.neurons_reshape.reshape(self.k,self.k)

    def test(self, x, country_names, categories):
        win = []
        x_standard = self.standard_i(x)
        for p in range(len(x)):
            winner_index = self.winner(x_standard[p],p)
            win.append(winner_index)
        print("test: ", x)
        print("categories: ",categories)
        for i in range(len(categories[0])):
            print("Categoria: ", categories[0][i])
            winner = win[i]
            print("Winner: ", country_names[winner], self.X[winner][i])

