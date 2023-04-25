import numpy as np
import json
import utils
import initdata
import perceptron

def main():

  with open('./config.json', 'r') as f:
    data = json.load(f)
    f.close()
  
  # define learning rate and epochs
  operation, learning_rate, epochs, bias = utils.getDataFromFile(data)

  # get data
  X = initdata.get_data()
  print(X)
  Y = initdata.get_data_expected(operation)
  print(Y)
  
  # initialize random weigths
  w = np.random.rand(len(X[0])+1)
  w[0]=bias
    
  for i in range(epochs):
    correct_predictions = 0
    # for each:
      # $O = Θ(\sum_{i=1}^n x_i*w_i - u)$
      # w^{nuevo} = w^{anterior} + η(ζ^µ - O^µ)x^µ 
    for j in range(len(X)):
      prediction = perceptron.predict(X[j],w)
      error = Y[j] - prediction
      w = perceptron.update_weigths(w,learning_rate,X[j], error)
      correct_predictions += int(prediction)
    print("pred correctas",correct_predictions)
    # calculate neuron error, if it converged done else go back to for

    accuracy = correct_predictions / (len(Y))
    print(f"Epoch {i+1}: Accuracy = {accuracy:.2f}")
    if accuracy >= 0.9:
        print("Stopping training. Desired accuracy reached.")
        break

  return None

if __name__ == '__main__':
  main()

  # {
  # "operation": "AND",
  # "learning_rate": 0.1,
  # "epochs": 100,
  # "bias": 1,
  # "operation_options": [
  # "AND", "XOR"
  # ]}