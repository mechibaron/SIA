import numpy as np
import csv
from sklearn.model_selection import train_test_split
#eta = 0.01 como mejor nos da
#eta = 0.1 muy malo 
#eta = 0.0001 mejor que 0.1 pero peor que 0.01

class LinearPerceptron:
    def __init__(self, learning_rate, input_size, epochs, bias=1,w=None):
        self.input_size = input_size
        self.bias = bias
        self.learning_rate = learning_rate
        self.epochs =epochs
        if(w is None):
            self.weights = np.random.rand(input_size+1)
        else:
            self.weights = w
        print(self.weights)

    def predict_linear(self,data):
        return np.dot(data, self.weights[1:]) + self.weights[0]
    
    def update_weigths_linear(self,x,error):
        self.weights[1:] += self.learning_rate * error *x
        self.weights[0] += self.learning_rate * error * self.bias

    def train_online(self, X, Z):
        error_by_epochs = []
        for i in range(self.epochs): 
            mse = 0
            outputs=[]
            for j in range(len(X)):
                O = self.predict_linear(X[j])
                error = Z[j] - O
                outputs.append(O)
                self.update_weigths_linear(X[j], error)
                # print("weigths: ", self.weights)
            mse = self.mean_square_error(Z, outputs)
            error_by_epochs.append(mse)
            
            print(f"Epoch {i+1}: Converged = {mse} ")

            # if(mse < 10 ): #Probando con tiradas vimos que aprende aprox hasta 9.5
            #     print("Stopping training. Converged.")
            #     break    
        return error_by_epochs       
            
                
    def mean_square_error(self, Z, output):
        mse = 0
        for i in range(len(Z)):
            mse += (Z[i] - output[i])**2
        mse /= len(Z)
        return mse
    
    def test(self, X_test, Z_test):
        outputs = []
        for inputs in zip(X_test):
            outputs.append(self.predict_linear(inputs))
        mse = self.mean_square_error(Z_test, outputs)
        print(f"Mean Square Error: {mse}")   
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
    X_train, X_test, Z_train, Z_test = train_test_split(X,Z,test_size=0.4, random_state=10)

    perc = LinearPerceptron(learning_rate, len(X[0]), epochs, bias)
    perc.train_online(X_train, Z_train)
    perc.test(X_test, Z_test)

if __name__ == '__main__':
    main()
