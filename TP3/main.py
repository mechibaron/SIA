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
  sourceData = initdata.get_data()
  print(sourceData)
  expectedData = initdata.get_data_expected(operation)

  # add bias
  for data in sourceData:
    data.insert(0,bias)
  
  # initialize random weigths
  w = np.random.rand(len(sourceData))
  print(w)
    
  for i in range(epochs):
    correct_predictions = 0
    # for each:
      # $O = Θ(\sum_{i=1}^n x_i*w_i - u)$
      # w^{nuevo} = w^{anterior} + η(ζ^µ - O^µ)x^µ 
    for j in range(len(sourceData)):
      prediction = perceptron.predict(sourceData[j],w)
      error = expectedData[j] - prediction
      w = perceptron.update_weigths(w,learning_rate,sourceData[j], error)
      correct_predictions += int(prediction == expectedData[j])
    
    # calculate neuron error, if it converged done else go back to for

    accuracy = correct_predictions / len(expectedData)
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