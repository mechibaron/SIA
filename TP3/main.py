import numpy as np
import json
import utils
import initdata
import perceptron
import plot1
def main():

  with open('./config.json', 'r') as f:
    data = json.load(f)
    f.close()
  
  # define learning rate and epochs
  # learning rate = eta (n)
  operation, learning_rate, epochs, bias = utils.getDataFromFile(data)

  # get data
  X = initdata.get_data()
  print(X)
  Z = initdata.get_data_expected(operation)
  print(Z)
  
  # initialize random weigths
  w = np.random.rand(len(X[0])+1)
  # w[0]=np.random.randint(-10,10)
  print(w[0])
  for i in range(epochs):
    correct_predictions = 0
    for j in range(len(X)):
      O = perceptron.predict(X[j],w)
      error = Z[j] - O
      # error = np.abs(Z[j] - O)
      w = perceptron.update_weigths(w,learning_rate,X[j], error, bias)
      # print("peso "+str(i) +": ",w)
      if(error==0):
        correct_predictions += 1
    print("Correct Predictions",correct_predictions)
    # calculate neuron error, if it converged done else go back to for

    accuracy = correct_predictions / (len(Z))
    print(f"Epoch {i+1}: Accuracy = {accuracy:.2f}")
    if accuracy >= 0.9:
      print("Stopping training. Desired accuracy reached.")
      break
  plot1.plot(w, operation, X, Z, i+1)


  return None

if __name__ == '__main__':
  main()
  