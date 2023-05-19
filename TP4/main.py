import json
import utils
from ej1 import ej1
from ej2 import ej2

if __name__ == '__main__':
  with open('./json/config.json', 'r') as f:
    data = json.load(f)
    f.close()

  epochs, exercise = utils.getDataFromFile(data)
  if exercise == 1:
    ej1(epochs)
  
  else:
    ej2(epochs)

    