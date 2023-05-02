import json
import utils

import step
import non_linear
import linear
import multicapa

if __name__ == '__main__':
  with open('./config.json', 'r') as f:
    data = json.load(f)
    f.close()
  # define learning rate and epochs
  # learning rate = eta (n)
  operation, learning_rate, epochs, bias, beta, type_perceptron, theta, item = utils.getDataFromFile(data)
  if(type_perceptron == 'escalon'):
    step.main(operation, learning_rate, epochs, bias)
  elif (type_perceptron == 'lineal'):
    linear.main(learning_rate, epochs, bias)
  elif (type_perceptron == 'no_lineal'):
    non_linear.main(learning_rate, epochs, bias, beta, theta)
  else:
    multicapa.main(learning_rate, epochs, bias, item)

    