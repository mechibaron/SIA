import numpy as np
import csv
from sklearn.model_selection import train_test_split



class NonlinearPerceptron:
    def __init__(self, learning_rate, epochs, input_size, theta='tanh', beta=1, bias=1):
        self.input_size = input_size
        self.bias = bias
        self.learning_rate = learning_rate
        self.epochs=epochs
        self.theta = theta
        self.beta = beta
        self.weights = np.random.rand(input_size+1)
        if(theta == 'tanh'):
            self.a = -1
        else:
            self.a = 0
        self.b = 1
    def tanh(self, x):
        return np.tanh(self.beta * x)

    def d_tanh(self, x):
        return self.beta * (1 - (self.tanh(x)) ** 2)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-2 * self.beta *x))

    def d_sigmoid(self, x):
        return 2*self.beta*self.sigmoid(x)*(1-self.sigmoid(x))
    
    def predict(self, inputs):
        x = np.dot(inputs, self.weights[1:])
        if self.theta == 'tanh':
            return self.tanh(x)
        else:
            return self.sigmoid(x)
        
    def update_weights(self, inputs, error):
        x = np.dot(inputs, self.weights[1:])
        if self.theta == 'tanh':
            self.weights[1:] += self.learning_rate * error *self.d_tanh(x)* inputs
            self.weights[0] += self.learning_rate * error *self.d_tanh(x)
        else:
            self.weights[1:] += self.learning_rate * error *self.d_sigmoid(x)* inputs
            self.weights[0] += self.learning_rate * error*self.d_sigmoid(x)
    
    def scaler(self, X, X_min, X_max):
        return ((X - X_min)/(X_max - X_min)*(self.b-self.a))+self.a
    
    def descaler(self, X_scaled, X_min, X_max):
        return ((X_scaled - self.a)/(self.b - self.a)*(X_max - X_min)) + X_min
    
    def train_online(self, X, Z):
        Z_min,Z_max = np.min(Z), np.max(Z)
        X1 = [arreglo[0] for arreglo in X]
        X2 = [arreglo[1] for arreglo in X]
        X3 = [arreglo[2] for arreglo in X]
        X1_min, X1_max = np.min(X1), np.max(X1)
        X2_min, X2_max = np.min(X2), np.max(X2)
        X3_min, X3_max = np.min(X3), np.max(X3)
        Z_scaled = self.scaler(Z, Z_min, Z_max)
        X1_scaled = self.scaler(X1, X1_min, X1_max)
        X2_scaled = self.scaler(X2, X2_min, X2_max)
        X3_scaled = self.scaler(X3, X3_min, X3_max)
        X_scaled = [[a, b, c] for a, b, c in zip(X1_scaled, X2_scaled, X3_scaled)]
        X_scaled = np.array(X_scaled)
        Z_scaled = np.array(Z_scaled)
        for epoch in range(self.epochs):
            outputs = []
            outputs_descaled = []
            for j in range(len(Z_scaled)):
                output= self.predict(X_scaled[j])
                output = np.array(output)
                error = Z_scaled[j] - output
                self.update_weights(X_scaled[j], error)
                outputs.append(output)
                output_descaled = self.descaler(output, Z_min, Z_max)                
                outputs_descaled.append(output_descaled)
                # print("esperado: ", Z[j], " obtenido: ", output_descaled)
            outputs_descaled = np.array(outputs_descaled)
            mse = np.sqrt(self.mean_square_error(Z, outputs_descaled))
            print(f"Epoch {epoch+1}: MSE = {mse}")
            if(self.theta == 'tanh'):
                if mse < 3: #Vimos con muchas pruebas que aprende hasta 2.8 aprox con un 70% del csv
                    print("Stopping training. Converged.")
                    break
            else:    
                if mse < 11: #Vimos con muchas pruebas que aprende hasta 10.28 aprox
                    print("Stopping training. Converged.")
                    break
    
    def test(self, X_test, Z_test):
        correct = 0
        for inputs, expected_output in zip(X_test, Z_test):
            output = self.predict(inputs)
            if np.abs(output - expected_output) < 10:
                correct += 1
        accuracy = correct / len(X_test)
        print(f"Test accuracy: {accuracy}")      

    def mean_square_error(self, Z, output):
        mse = 0
        for i in range(len(Z)):
            mse += (Z[i] - output[i])**2
        mse /= len(Z)
        return mse
    

def main(learning_rate, epochs, bias, beta, theta):
    X = []
    Z = []
    with open('./data/TP3-ej2-conjunto.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            X.append([float(row['x1']), float(row['x2']), float(row['x3'])])
            Z.append(float(row['y']))
    X = np.array(X)
    Z = np.array(Z)
    X_train, X_test, Z_train, Z_test = train_test_split(X, Z, test_size=0.3, random_state=42)

    perc = NonlinearPerceptron(learning_rate, epochs, len(X[0]), theta, beta, bias)
    perc.train_online(X_train, Z_train)
    perc.test(X_test, Z_test)

if __name__ == '__main__':
    main()
