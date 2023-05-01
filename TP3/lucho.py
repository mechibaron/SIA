import numpy as np
import csv
import perceptron
from sklearn.preprocessing import StandardScaler



class NonlinearPerceptron:
    def __init__(self, input_size, learning_rate, theta='tanh', beta=1):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.theta = theta
        self.beta = beta
        self.weights = np.random.rand(input_size+1)
        
    def predict(self, inputs):
        x = np.dot(inputs, self.weights[1:])
        if self.theta == 'tanh':
            return np.tanh(self.beta*x)
        else:
            return 1 / (1 + np.exp(-self.beta*x))
        
    def update_weights(self, inputs, error):
        self.weights[1:] += self.learning_rate * error * inputs
        self.weights[0] += self.learning_rate * error
        
    def train_online(self, X, Z, epochs):
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        scaler_Z = StandardScaler()
        Z_scaled = scaler_Z.fit_transform(Z.reshape(-1, 1)).reshape(-1)
        for epoch in range(epochs):
            for inputs, expected_output in zip(X_scaled, Z_scaled):
                output = self.predict(inputs)
                error = expected_output - output
                self.update_weights(inputs, error)
                z_exp_descaled = scaler_Z[inputs].inverse_transform(expected_output)
                z_descaled = scaler_X.inverse_transform(output)
                print("esperado: ", z_exp_descaled, " obtenido: ", z_descaled)
            mse = self.mean_square_error(X, Z)
            print(f"Epoch {epoch+1}: MSE = {mse}")
            if mse < 0.1:
                print("Stopping training. Converged.")
                break
                
                
    def mean_square_error(self, X, Z):
        mse = 0
        for inputs, expected_output in zip(X, Z):
            output = self.predict(inputs)
            mse += 0.5 * (expected_output - output)**2
        mse /= len(X)
        return mse

def main():
    X = []
    Z = []
    with open('./data/TP3-ej2-conjunto.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            X.append([float(row['x1']), float(row['x2']), float(row['x3'])])
            Z.append(float(row['y']))
    X = np.array(X)
    Z = np.array(Z)
    perc = NonlinearPerceptron(input_size=3, learning_rate=0.1, theta='tanh', beta=1)
    perc.train_online(X, Z, epochs=100)

if __name__ == '__main__':
    main()
