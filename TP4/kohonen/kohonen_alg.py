import numpy as np

class Kohonen:

    def __init__(self,p, n, k,radio, learning_rate, similitud, epochs, X):
        self.p = p
        self.n = n
        # nostras pensamos => k = n
        self.k = k
        self.weights = np.random.rand(k**2,n)
        self.radio = [radio,radio]
        self.learning_rate = [learning_rate,learning_rate]
        self.similitud = similitud
        self.epoch = epochs
        self.X = X
    

    def euclidea (self, x,p):
        w=[]
        for j in range(self.k): #recorriendo filas
            w.append(np.abs(x - self.weights[j][p]))
        wk = min(w)
        return w.index(wk)
    
    def exponencial (self, x, p):
        print(self.weights)
        w=[]
        for j in range(self.k): #recorriendo filas
            w.append(np.exp(-((x - self.weights[j][p])**2)))
        wk = min(w)
        return w.index(wk)
    
    def regla_de_kohonen(self, winner_index,x, p):
        # print("REGLA DE KOHENEN")
        for j in range(self.k**2):
            if(np.abs(j-winner_index) <= self.radio[1]):
                # Si soy vecino actulizo mis pesos
                self.weights[j][p] += self.learning_rate[1] * (x-self.weights[j][p])  
        
    def standard(self, X):
        X_standard = []
        for i in range(len(X)):
            X_standard.append(self.standard_i(X[i]))    
        return X_standard
    
    def standard_i(self,x):
        mean =[np.mean(x) for _ in range(len(x))]
        std = [np.std(x) for _ in range(len(x))]
        return (x - np.array(mean))/np.array(std)    
    
    def winner(self, x,p):
        if(self.similitud == "euclides"):
            return self.euclidea(x,p)
        else:
            return self.exponencial(x,p)
        
    def train_kohonen(self):
        X_standard = self.standard(self.X)
        # X_standard = X
        for i in range(self.epoch):
            for j in range(len(X_standard)):
                for p in range(len(X_standard[j])):
                    # Seleccionar un registro de entrada X^p
                    x = X_standard[j][p]
                    # Encontrar la neurona ganadora
                    winner_index = self.winner(x,p)
                    # Actualizar los pesos segun kohonen
                    self.regla_de_kohonen(winner_index, x, p)
                
            # Ajuste de radio:
            ajuste = self.radio[0] * (1 - i/self.epoch)
            self.radio[1] = 1 if ajuste < 1 else ajuste
            # Ajuste de ETA:
            self.learning_rate[1] = self.learning_rate[0] * (1 - i/self.epoch)
            
            # Segunda opcion de ajuste de ETA mas abruta (para ppt probar la dif entre las dos)
            # decay_rate es un hiperparametro que recibiriamos.
            # self.learning_rate[1] = self.learning_rate[0] * np.exp(-decay_rate*i)

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

