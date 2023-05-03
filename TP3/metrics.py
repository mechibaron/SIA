import random
from multilayer.multilayer_perceptron import MultilayerPerceptron
import numpy as np
from constants import *


def accuracy(confusion_matrix, matrix_dim, element_position):
    right_ans = confusion_matrix[element_position][element_position]
    wrong_ans = 0
    for i in range(matrix_dim):
        for j in range(matrix_dim):
            if i != element_position and j != element_position:
                wrong_ans += confusion_matrix[i][j]
    return right_ans / wrong_ans


def precision(confusion_matrix, matrix_dim, element_position):
    true_positives = confusion_matrix[element_position][element_position]
    total_positives = 0
    for j in range(matrix_dim):
        total_positives += matrix_dim[element_position][j]
    return true_positives / total_positives


def recall(confusion_matrix, matrix_dim, element_position):
    true_positives = confusion_matrix[element_position][element_position]
    real_positives = 0
    for i in range(matrix_dim):
        real_positives += confusion_matrix[i][element_position]
    return true_positives / real_positives


def f1_score(confusion_matrix, matrix_dim, element_position):
    precision_value = precision(confusion_matrix, matrix_dim, element_position)
    recall_value = recall(confusion_matrix, matrix_dim, element_position)
    return 2 * precision_value * recall_value / (precision_value + recall_value)


def cross_validation(k, training_set, expected_output, perceptron_type, amount, learning_rate, batch_size=1,
                     learning_rate_params=None, momentum=False):

    if not (len(training_set) % k == 0):
        print("Choose another partition size")
        exit()

    all_indexes = list(range(len(training_set)))
    random.shuffle(all_indexes)
    split_indexes = np.array_split(np.array(all_indexes), k)

    best_result = float('inf')
    best_network = None
    best_indexes = None

    for indexes in split_indexes:
        training_set_idx = set(all_indexes) - set(indexes)

        sub_training_set = [training_set[i] for i in training_set_idx]
        sub_expected_output = [expected_output[i] for i in training_set_idx]
        test_set = [training_set[i] for i in indexes]
        test_output = [expected_output[i] for i in indexes]

        # if perceptron_type == LINEAR:
        #     perceptron = LinearPerceptron(sub_training_set, sub_expected_output, learning_rate)
        # elif perceptron_type == NON_LINEAR:
        #     perceptron = NonLinearPerceptron(sub_training_set, sub_expected_output, learning_rate)
        # else:
        perceptron = MultilayerPerceptron(sub_training_set, sub_expected_output, learning_rate, batch_size,
                                              learning_rate_params, momentum)

        perceptron.train(amount)

        res = perceptron.test_input(test_set)
        acc = get_accuracy(res, test_output, criteria=lambda x, y: np.abs(x - y) < 0.1)
        print("ACCURACY: \n", acc)

        if acc < best_result:
            best_result = acc
            best_network = perceptron
            best_indexes = indexes

    return best_result, best_network, best_indexes


# Ejercicio 1 - XOR: 1 -1
# Ejercicio 2 - par/impar: par = 1, impar = -1
# Ejercicio 3 - digito: 1 0
def get_confusion_matrix(classes, real_output, expected_output):
    matrix = [[0, 0], [0, 0]]
    for i in range(len(real_output)):
        if real_output[i] == expected_output[i]:
            if real_output[i] == classes[0]:
                matrix[0][0] += 1
            else:
                matrix[1][1] += 1
        else:
            if real_output == classes[0]:
                matrix[1][0] += 1
            else:
                matrix[0][1] += 1
    return matrix


def get_accuracy(real_output, expected_output, criteria=None):
    results = np.zeros(2)
    for i in range(len(real_output)):
        if criteria is not None:
            right_answer = criteria(real_output[i], expected_output[i])
        else:
            right_answer = real_output[i] == expected_output[i]
        if right_answer:
            results[0] += 1
        else:
            results[1] += 1
    right_answers = results[0]
    wrong_answers = results[1]
    return right_answers / (right_answers + wrong_answers)


def get_metrics(results):
    print(f'Aciertos: {results[0]}')
    print(f'Errores: {results[1]}')
    print(f'Accuracy: {accuracy(results)}')
