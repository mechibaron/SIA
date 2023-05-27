import numpy as np
import utils
from hopfield import plots, hopfield
import json

def ej2(epochs):
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
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
        title = "Letter " + letters[letter_idx]
        # plots.plot_letter(matrix_letters[letter_idx], title)

    # Comparar ortogonalidad en training
    ortogonalidad = []
    for i in train_letters_idx:
        for j in train_letters_idx:
            if(i!=j):
                letter_1=matrix_letters[i].flatten()
                letter_2=matrix_letters[j].flatten()
                norm_1=np.linalg.norm(letter_1)
                norm_2=np.linalg.norm(letter_2)
                cos_tita=(np.dot(letter_1,letter_2))/(norm_1*norm_2)
                arc_cos_tita = np.arccos(cos_tita)
                ortogonalidad.append(arc_cos_tita)
    print("Ortogonalidad between ", train_letters, sum(ortogonalidad)/len(ortogonalidad))

    # Inicializo con matrices de entrenamiento
    model = hopfield.Hopfield(epochs,5, train_matrix)
    iterations = 1
    positive = 0
    fake_positive = 0
    negative = 0
    for _ in range(iterations):
        # Noisy letters
        noise_idx = get_letter_idx(letters, noisy_letter)
        noise_letter = create_noise(matrix_letters[noise_idx], noise_probability)
        # Idx del arreglo de entrenamiento al que pertenece
        train_noise_idx = get_letter_idx(train_letters, noisy_letter)
        # Print de letras noisy
        plots.plot_letter(noise_letter, "Noisy Letter")

        # Devuelve el estado al que llego, response e idx
        response, state, idx = model.train(noise_letter)
       
        if (response == True):
            if(idx == train_noise_idx):
                # Entonces devuelve la letra a la que se aplico el ruido
                positive+=1
            else:
                # Entonces devuelve una letra del entrenamiento que no es la que le aplico ruido
                fake_positive+=1
        else:
            # Es otra cosa que no pertenece al train
            negative+=1

        plots.plot_letter(state, "Final State")
    print("For probability: ", noise_probability)
    print("Positivie: ", positive)
    print("Fake Positivie: ", fake_positive)
    print("Negative: ", negative)
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
    if letter in letters:
        return letters.index(letter)
    else: 
        return -1
