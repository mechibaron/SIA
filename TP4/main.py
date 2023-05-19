import json
import utils
import numpy as np
from ej1 import ej1

if __name__ == '__main__':
  with open('./json/config.json', 'r') as f:
    data = json.load(f)
    f.close()
  # define learning rate and epochs
  # learning rate = eta (n)
  learning_rate, epochs, exercise = utils.getDataFromFile(data)
  if exercise == 1:
    with open('./json/config_ej1.json', 'r') as f:
      ej1_data = json.load(f)
      f.close()
    type_model, similitud, radio, k = utils.getDataFromEj1(ej1_data)
    ej1(learning_rate, epochs, type_model, similitud, radio, k)
  
  # else:
    # ej2()

    