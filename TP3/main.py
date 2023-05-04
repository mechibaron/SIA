import json
import utils
import numpy as np
import step
import non_linear
import linear
from ej3_1 import ej1
from ej3_2 import ej2
from ej3_3 import ej3


if __name__ == '__main__':
  with open('./json/config.json', 'r') as f:
    data = json.load(f)
    f.close()
  # define learning rate and epochs
  # learning rate = eta (n)
  learning_rate, epochs, bias, type_perceptron = utils.getDataFromFile(data)
  if(type_perceptron == 'escalon'):
    with open('./json/ej1_config.json', 'r') as f:
      ej1 = json.load(f)
      f.close()
    operation = utils.getDataForEj1(ej1)
    step.main(operation, learning_rate, epochs, bias)
  elif (type_perceptron == 'lineal'):
    linear.main(learning_rate, epochs, bias)
  elif (type_perceptron == 'no_lineal'):
    with open('./json/ej2_config.json', 'r') as f:
      ej2 = json.load(f)
      f.close()
    beta, theta = utils.getDataForEj2(ej2)
    non_linear.main(learning_rate, epochs, bias, beta, theta)
  else:
    with open("./json/ej3_config.json") as jsonFile:
      jsonObject = json.load(jsonFile)
      jsonFile.close()

    exercise = int(jsonObject["exercise"])
    exerciseFile = str(jsonObject["config_file"][exercise])

    with open(exerciseFile) as jsonFile:
        configJsonObject = json.load(jsonFile)
        jsonFile.close()

    hiddenLayers = np.array(configJsonObject["hiddenLayers"])
    batch_size = configJsonObject["batch_size"]
    momentum = configJsonObject["momentum"]
    adaptive_eta = configJsonObject["adaptive_eta"]
    set_momentum = False
    adaptive_params = None

    if adaptive_eta == 1:
        adaptive_k = int(configJsonObject["adaptive_k"])
        adaptive_inc = int(configJsonObject["adaptive_inc"])
        adaptive_dec = int(configJsonObject["adaptive_dec"])
        adaptive_params = [adaptive_inc, adaptive_dec, adaptive_k]

    if momentum == 1:
        set_momentum = True

    if exercise == 0:
        ej1(learning_rate, epochs, hiddenLayers, batch_size, set_momentum, adaptive_params)

    elif exercise == 1:
        ej2(learning_rate, epochs, hiddenLayers, batch_size, set_momentum, adaptive_params)
        
    else:
        ej3(learning_rate, epochs, hiddenLayers, batch_size, set_momentum, adaptive_params)


    