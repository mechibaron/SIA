import numpy as np
import initdata
import plot1

class StepPerceptron:
    def __init__(self, operation, learning_rate, epochs, bias, input_size):
        self.input_size = input_size
        self.bias = bias
        self.operation = operation
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.random.rand(input_size+1)

    def step_function(self,x):
        if x < 0:
            return -1
        else:  
            return 1
    def update_weigths(self,sourceData,error):
        for i in range(1,len(self.weights)):
            self.weights[i] += self.learning_rate * error * sourceData[i-1]

        self.weights[0] += self.learning_rate * error * self.bias

    def predict(self, data):
        value = np.dot(data, self.weights[1:]) + self.weights[0]
        print("Predict value: ", value)
        return self.step_function(value)
    
    def train_online(self, X, Z):
        print("EPOCAS", self.epochs)

        for i in range(self.epochs):
            correct_predictions = 0
            for j in range(len(X)):
                O = self.predict(X[j])
                error = Z[j] - O
                self.update_weigths(X[j], error)
                if(error==0):
                    correct_predictions += 1
                print("Correct Predictions",correct_predictions)
                # calculate neuron error, if it converged done else go back to for

            accuracy = correct_predictions / (len(Z))
            print(f"Epoch {i+1}: Accuracy = {accuracy:.2f}")
            if accuracy >= 0.9:
                print("Stopping training. Desired accuracy reached.")
                break
        plot1.plot(self.weights, self.operation, X, Z, i+1)


def main(operation, learning_rate, epochs, bias):
    X = initdata.get_data()
    Z = initdata.get_data_expected(operation)   
    perc = StepPerceptron(operation, learning_rate, epochs,bias, len(X[0]))
    perc.train_online(X, Z)

if __name__ == '__main__':
    main()
