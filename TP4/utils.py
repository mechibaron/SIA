import csv
import numpy as np
from constants import FIRST, LAST, MIDDLE
from multilayer.multilayer_perceptron import MultilayerPerceptron


def getDataFromFile(data):
    return data['learning_rate'], data['epochs'], data['exercise'], data['type_model']

def getDataFromEj1(data):
    return data['similitud'], data['radio'], data['k']

def getDataFromEj2(data):
    return data['train_letters']

def import_letters_data(file, quantity):
    csv_file = open(file, 'r', newline='\n')
    csv_reader = csv.reader(csv_file, delimiter=" ")
    # Arreglo donde van a estar las 25 matrices (1 por letra)
    entry = []
    row_count = 0
    matrix = []
    matrix = np.empty((quantity,quantity))
    data = []
    for row in csv_reader:
        for a in row:
            if a != '':
                data.append(a)
        matrix[row_count] = data
        data = []
        row_count += 1
        if row_count == quantity:
            entry.append(matrix)
            matrix = np.empty((quantity,quantity))
            row_count = 0
    return entry

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
