import numpy as np
import json
import csv
import utils
import initdata
import perceptron
import plot1
def main1(operation, learning_rate, epochs, bias):
  # get data
  X = initdata.get_data()
  print(X)
  Z = initdata.get_data_expected(operation)
  print(Z)
  
  # initialize random weigths
  w = np.random.rand(len(X[0])+1)
  print(w[0])
  for i in range(epochs):
    correct_predictions = 0
    for j in range(len(X)):
      O = perceptron.predict(X[j],w)
      error = Z[j] - O
      w = perceptron.update_weigths(w,learning_rate,X[j], error, bias)
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

def main2_lineal(learning_rate, epochs, bias):
  # min_out=0
  # max_out=100
  X = []
  Z = []
  with open('./data/TP3-ej2-conjunto.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      X.append([row['x1'], row['x2'], row['x3']])
      Z.append(row['y'])
  X = np.array(X, dtype=float)
  Z = np.array(Z, dtype=float)
  w = np.random.rand(len(X[0])+1)
  print("w: ",w)
  for i in range(epochs):
    mse = 0
    error=[]
    for j in range(len(X)):
      O = perceptron.predict_linear(X[j],w)
      # print("O: ",O)
      # scaled_O=perceptron.scaled_lineal(O, min_out, max_out)
      error.append((1/2)(Z[j] - O)**2)
      mse += perceptron.mean_square_error(Z[j], O)
    
    w = perceptron.update_weigths_linear(w,learning_rate,X, error, bias)
    print("weigths: ", w)
    mse = mse / (j+1)
    converged = np.sqrt(mse)
    print(f"Epoch {i+1}: Converged = {converged} , MSE = {mse}")

    if(-10 < converged < 10 ):
      print("Stopping training. Converged.")
      break
    

  return None

def main2_no_lineal(learning_rate, epochs, bias, beta, theta):
  X = []
  Z = []
  with open('./data/TP3-ej2-conjunto.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      X.append([row['x1'], row['x2'], row['x3']])
      Z.append(row['y'])
  if(theta=="tanh"):
    interval = (-1,1)
  else:
    interval = (0,1)
  X = np.array(X, dtype=float)
  Z = np.array(Z, dtype=float)
  w = np.random.rand(len(X[0])+1)
  Z_norm= perceptron.escale_all(Z,interval)
  print(Z)
  for i in range(epochs):
    mse=0
    for j in range(len(X)):
      X_p = np.dot(X[j] , w[1:])
      if(theta=="tanh"):
        O = perceptron.tanh(X_p,beta)
      else:
        O = perceptron.sigmoid(X_p,beta)
      error=((1/2)*(Z_norm[j] - O)**2)
      # mse += perceptron.mean_square_error(Z[j], O)
      mse += perceptron.mean_square_error(Z[j], perceptron.denormalize(Z, O))
      print(j,"Expected: ", Z[j], "Obtained: ", perceptron.denormalize(Z,O))
      w = perceptron.update_weigths_no_linear(w,learning_rate, error, X[j] , bias, theta, beta, O)
    mse = mse / (j+1)
    converged = np.sqrt(mse)
    print(f"Epoch {i+1}: Converged = {converged} , MSE = {mse}")

    if( -10 < converged < 10 ):
      print("Stopping training. Converged.")
      break     

if __name__ == '__main__':
  with open('./config.json', 'r') as f:
    data = json.load(f)
    f.close()
  
  # define learning rate and epochs
  # learning rate = eta (n)
  operation, learning_rate, epochs, bias, beta, type_perceptron, theta = utils.getDataFromFile(data)
  if(type_perceptron == 'escalon'):
    main1(operation, learning_rate, epochs, bias)
  elif (type_perceptron == 'lineal'):
    main2_lineal(learning_rate, epochs, bias)
  elif (type_perceptron == 'no_lineal'):
    main2_no_lineal(learning_rate, epochs, bias, beta, theta)
  # else:
    # main3
    