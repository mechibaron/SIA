import numpy as np
import utils
from hopfield import plots, hopfield
import json

def ej2(epochs):
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    matrix_letters = utils.import_letters_data('./data/letters_matrix', 5)
    with open('./json/config_hopfield.json', 'r') as f:
        ej2_data = json.load(f)
        f.close()
    train_letters, noisy_letter, noise_probability = utils.getDataFromEj2(ej2_data)

    train_letters_idx = []
    for letter in train_letters:
        if letter in letters:
            index = get_letter_idx(letters,letter)
            train_letters_idx.append(index)

    # Print de letras de testeo
    train_matrix = []
    for letter_idx in train_letters_idx:
        train_matrix.append(matrix_letters[letter_idx])
        # plots.plot_letter(matrix_letters[letter_idx])
        
    # Inicializo con matrices de entrenamiento
    model = hopfield.Hopfield(epochs,5, train_matrix)
    
    # Noisy letters
    noise_idx = get_letter_idx(letters, noisy_letter)
    noise_letter = create_noise(matrix_letters[noise_idx], noise_probability)

    # Print de letras noisy
    plots.plot_letter(noise_letter)

    # Devuelve el estado al que llego
    state = model.train(noise_letter)
    plots.plot_letter(state)

    return None


def create_noise(matrix, noise_probability):
    test_set = np.zeros((len(matrix), len(matrix[0])))
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            test_set[i][j] = noise(matrix[i][j], noise_probability)
    return test_set

def noise(number, noise_probability):
    probability = np.random.rand(1)[0]
    if probability < noise_probability:
        if number == 1:
            return -1
        else:
            return 1
    return number

def get_letter_idx(letters, letter):
    return letters.index(letter)
