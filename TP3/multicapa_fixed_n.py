import numpy as np

class NeuralNetwork:
    
    def __init__(self, input_dim, output_dim, hidden_dim, hidden_layers):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.weights = []
        self.bias = []
        for i in range(hidden_layers+1):
            if i == 0:
                self.weights.append(np.random.randn(input_dim, hidden_dim))
                self.bias.append(np.zeros((1, hidden_dim)))
            elif i == hidden_layers:
                self.weights.append(np.random.randn(hidden_dim, output_dim))
                self.bias.append(np.zeros((1, output_dim)))
            else:
                self.weights.append(np.random.randn(hidden_dim, hidden_dim))
                self.bias.append(np.zeros((1, hidden_dim)))
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def step(self, z):
        return np.where(z >= 0, 1, -1)
    
    def forward_propagation(self, x):
        a = []
        z = []
        for i in range(self.hidden_layers+1):
            if i == 0:
                z.append(np.dot(x, self.weights[i]) + self.bias[i])
                a.append(self.step(z[i]))
            elif i == self.hidden_layers:
                z.append(np.dot(a[i-1], self.weights[i]) + self.bias[i])
                a.append(self.step(z[i]))
            else:
                z.append(np.dot(a[i-1], self.weights[i]) + self.bias[i])
                a.append(self.step(z[i]))
        return a
    
    def backward_propagation(self, x, y, a):
        dw = []
        db = []
        delta = []
        for i in range(self.hidden_layers+1)[::-1]:
            if i == self.hidden_layers:
                delta.append(a[i] - y)
                dw.append(np.dot(a[i-1].T, delta[-1]))
                db.append(np.sum(delta[-1], axis=0, keepdims=True))
            elif i == 0:
                delta.append(np.dot(delta[-1], self.weights[i+1].T) * (a[i] * (1-a[i])))
                dw.append(np.dot(x.reshape(-1, 1), delta[-1].reshape(1, -1)))
                db.append(np.sum(delta[-1], axis=0, keepdims=True))
            else:
                delta.append(np.dot(delta[-1], self.weights[i+1].T) * (a[i] * (1-a[i])))
                dw.append(np.dot(a[i-1].T, delta[-1]))
                db.append(np.sum(delta[-1], axis=0, keepdims=True))
        return dw[::-1], db[::-1]
    
    def update_weights(self, dw, db, learning_rate):
        for i in range(self.hidden_layers+1):
            self.weights[i] -= learning_rate * dw[i]
            self.bias[i] -= learning_rate * db[i]
    
    def train(self, x, y, epochs, learning_rate):
        for epochs in range(epochs):
            for i in range(len(x)):
                a = self.forward_propagation(x[i])
                dw, db = self.backward_propagation(x[i], y[i], a)
                self.update_weights(dw, db, learning_rate)
            if(epochs==2):
                print(a)
    
    def predict(self, x):
        a = self.forward_propagation(x)
        return a[-1]
        
def main():
    # Definimos los datos de entrada y salida
    X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    Y = np.array([[-1], [-1], [-1], [1]])

    # Definimos los parámetros de la red
    input_dim = 2
    output_dim = 1
    hidden_dim = 7
    hidden_layers = 5
    epochs=1000
    learning_rate=0.1
    # Creamos la red neuronal
    nn = NeuralNetwork(input_dim, output_dim, hidden_dim, hidden_layers)
    nn.train(X, Y, epochs, learning_rate)
    for x, y in zip(X, Y):
        predicted = nn.predict(x)
        print(f"Input: {x} - Output: {predicted} - Target: {y}")
if __name__ == '__main__':
    main()