import numpy as np
import utils
import matplotlib.pyplot as plt
from hopfield import plots
import json

def ej2(learning_rate, epochs):
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    input_letters = utils.import_letters_data('./data/letters_matrix', 5)
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
        plots.plot_letter(input_letters[letter_idx])

    return None

def get_letter_idx(letters, letter):
    return letters.index(letter)
