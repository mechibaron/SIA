import numpy as np
import utils
import matplotlib.pyplot as plt
from hopfield import plots
import json

def ej2(learning_rate, epochs):
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    matrix_letters = utils.import_letters_data('./data/letters_matrix', 5)
    with open('./json/config_hopfield.json', 'r') as f:
        ej2_data = json.load(f)
        f.close()
    train_letters = utils.getDataFromEj2(ej2_data)

    train_letters_idx = []
    for letter in train_letters:
        if letter in letters:
            index = get_letter_idx(letters,letter)
            train_letters_idx.append(index)

    # Print de letras de testeo
    for letter_idx in train_letters_idx:
        plots.plot_letter(matrix_letters[letter_idx])
    
    # Noisy letters
    for noise_idx in train_letters_idx:
        noise_letter = create_noise(matrix_letters[noise_idx])
        # Print de letras noisy
        plots.plot_letter(noise_letter)


    return None


def create_noise(test_set):
    for i in range(len(test_set)):
        for j in range(len(test_set[i])):
            test_set[i][j] = noise(test_set[i][j])
    return test_set


def noise(number):
    probability = np.random.rand(1)[0]
    print(probability)
    if probability < 0.1:
        if number == 1:
            return -1
        else:
            return 1
    return number

def get_letter_idx(letters, letter):
    return letters.index(letter)
