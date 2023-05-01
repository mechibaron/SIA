import numpy as np
import json
import csv
import utils
import perceptron

import step
import non_linear

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
    # error=[]
    for j in range(len(X)):
      O = perceptron.predict_linear(X[j],w)
      # print("O: ",O)
      # scaled_O=perceptron.scaled_lineal(O, min_out, max_out)
      error = Z[j] - O
      w = perceptron.update_weigths_linear(w,learning_rate,X[j], error, bias)
      # error.append((1/2)(Z[j] - O)**2)
      mse += perceptron.mean_square_error(Z[j], O)
    
    # w = perceptron.update_weigths_linear(w,learning_rate,X, error, bias)
    print("weigths: ", w)
    mse = mse / (j+1)
    converged = np.sqrt(mse)
    print(f"Epoch {i+1}: Converged = {converged} , MSE = {mse}")

    if(-10 < converged < 10 ):
      print("Stopping training. Converged.")
      break
    

  return None

if __name__ == '__main__':
  with open('./config.json', 'r') as f:
    data = json.load(f)
    f.close()
  
  # define learning rate and epochs
  # learning rate = eta (n)
  operation, learning_rate, epochs, bias, beta, type_perceptron, theta = utils.getDataFromFile(data)
  if(type_perceptron == 'escalon'):
    step.main(operation, learning_rate, epochs, bias)
  elif (type_perceptron == 'lineal'):
    main2_lineal(learning_rate, epochs, bias)
  elif (type_perceptron == 'no_lineal'):
    non_linear.main(learning_rate, epochs, bias, beta, theta)
  # else:
    # main3
    