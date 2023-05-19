import csv
import numpy as np
from constants import FIRST, LAST, MIDDLE
from multilayer.multilayer_perceptron import MultilayerPerceptron

#def getDataFromFile(data):
#    return data['learning_rate'], data['epochs'], data['exercise']
def getDataFromFile(data):
    return data['learning_rate'], data['epochs']

def getDataFromEj1(data):
    return data['type_model'], data['similitud'], data['radio'], data['k']

# def normalize(output):
#     min_expected = min(output)
#     max_expected = max(output)
#     return np.array(list(map(lambda x: 2 * ((x - min_expected) / (max_expected - min_expected)) - 1, output)))


# def import_data(file, quantity):
#     csv_file = open(file, 'r')
#     csv_reader = csv.reader(csv_file, delimiter=" ")
#     data = []
#     entry = []
#     row_count = 0
#     for row in csv_reader:
#         if quantity == 1:
#             entry = [float(a) for a in row if a != '']
#             data.append(entry)
#         else:
#             row_count += 1
#             for a in row:
#                 if a != '':
#                     entry.append(float(a))
#             if row_count == quantity:
#                 data.append(entry)
#                 entry = []
#                 row_count = 0
#     return data

def import_data(file):
    csv_file = open(file, 'r')
    csv_reader = csv.reader(csv_file, delimiter=",")
    data = []
    names = []
    categories = []
    for i in csv_reader:
        if(i[0] == "Country"):
            categories.append(i[1:])
        else:
            names.append(i[0])
            data.append(i[1:])
    return names, data, categories[0]




# def create_multilayer_perceptron_and_train(training_set, expected_output, learning_rate, epochs, layers, batch_size, momentum=False, adaptive_params=None):
#     perceptron = MultilayerPerceptron(training_set, expected_output, learning_rate, adaptive_params, batch_size, momentum)
#     perceptron.add(len(training_set[0]), FIRST)
#     for i in range(len(layers)):
#         perceptron.add(layers[i], MIDDLE)
#     perceptron.add(len(expected_output[0]), LAST)
#     perceptron.train(epochs)
#     return perceptron
