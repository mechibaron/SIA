import numpy as np
import csv


class LinearPerceptron:
    def __init__(self, learning_rate, input_size, epochs, bias=1):
        self.input_size = input_size
        self.bias = bias
        self.learning_rate = learning_rate
        self.epochs =epochs
        self.weights = np.random.rand(input_size+1)

    def predict_linear(self,data):
        return np.dot(data, self.weights[1:]) + self.weights[0]
    
    def update_weigths_linear(self,x,error):
        self.weights[1:] += self.learning_rate * error *x
        self.weights[0] += self.learning_rate * error * self.bias

    def train_online(self, X, Z):
        for i in range(self.epochs):
            mse = 0
            outputs=[]
            for j in range(len(X)):
                O = self.predict_linear(X[j])
                error = Z[j] - O
                outputs.append(O)
                self.update_weigths_linear(X[j], error)
                # print("weigths: ", self.weights)
            mse = np.sqrt(self.mean_square_error(Z, outputs))
            
            print(f"Epoch {i+1}: Converged = {mse} ")

            if(mse < 10 ):
                print("Stopping training. Converged.")
                break    
        return None       
            
                
    def mean_square_error(self, Z, output):
        mse = 0
        for i in range(len(Z)):
            mse += (Z[i] - output[i])**2
        mse /= len(Z)
        return mse
    

def main(learning_rate, epochs, bias):
    X = []
    Z = []
    with open('./data/TP3-ej2-conjunto.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            X.append([float(row['x1']), float(row['x2']), float(row['x3'])])
            Z.append(float(row['y']))
    X = np.array(X)
    Z = np.array(Z)
    perc = LinearPerceptron(learning_rate, len(X[0]), epochs, bias)
    perc.train_online(X, Z)

if __name__ == '__main__':
    main()
