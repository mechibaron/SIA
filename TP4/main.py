import json
import utils
from ej1 import ej1

if __name__ == '__main__':
  with open('./json/config.json', 'r') as f:
    data = json.load(f)
    f.close()

  learning_rate, epochs, exercise, type_model = utils.getDataFromFile(data)
  if exercise == 1:
    ej1(learning_rate, epochs, type_model)
  
  # else:
    # ej2()

    