import numpy as np
import ej3_utils


def ej2(learning_rate, epochs, layers, batch_size, momentum=False, adaptive_params=None):
    inputs = ej3_utils.import_data('data/ej3_23_training_set', 7)
    outputs = np.array(ej3_utils.import_data('data/ej3_2_expected_output', 1), dtype=float)

    training_set = np.array(inputs)
    expected_output = np.array(outputs)

    perceptron = ej3_utils.create_multilayer_perceptron_and_train(training_set, expected_output, learning_rate, epochs, layers, batch_size, momentum, adaptive_params)
    results = np.array(perceptron.test_input(training_set), dtype=float)
    print('Expected      Result')

    for i in range(results.size):
        print(f'{np.round(expected_output[i])[0]}\t \t\t   {np.abs(np.round(results[i])[0])}')
