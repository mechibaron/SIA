import json
import numpy as np

from ej3_1 import ex1
from ej3_2 import ex2
from ej3_3 import ex3

with open("ex3_config.json") as jsonFile:
    jsonObject = json.load(jsonFile)
    jsonFile.close()

exercise = int(jsonObject["exercise"])
exerciseFile = str(jsonObject["config_file"][exercise])

with open(exerciseFile) as jsonFile:
    configJsonObject = json.load(jsonFile)
    jsonFile.close()

learning_rate = configJsonObject["learning_rate"]
epochs = configJsonObject["epochs"]
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
    ex1(learning_rate, epochs, hiddenLayers, batch_size, set_momentum, adaptive_params)
elif exercise == 1:
    ex2(learning_rate, epochs, hiddenLayers, batch_size, set_momentum, adaptive_params)
else:
    ex3(learning_rate, epochs, hiddenLayers, batch_size, set_momentum, adaptive_params)


# import the data
# create the model based on the data entry and the config.json
# train
# show the results, metrics and plots
