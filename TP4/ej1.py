import numpy as np

import utils
from constants import FIRST, MIDDLE, LAST
from kohonen import kohonen_alg

def ej1(learning_rate, epochs, type_model, similitud, radio):
    input_names, inputs, categories = utils.import_data('data/europe.csv')
    
    country_name_train = np.array(input_names)
    country_name_train = np.delete(country_name_train,3)
    
    training_set = np.array(inputs, dtype=float)
    if(type_model == 'kohonen'):
        model = kohonen_alg.Kohonen(len(training_set) -1, len(training_set[0]), len(training_set[0]), radio, learning_rate, similitud, epochs,np.delete(training_set,3, axis=0))
        model.train_kohonen()
    # else:
    #  modelo de oja
     
     
    model.test(training_set[3], country_name_train, categories)

   
    # results = np.array(perceptron.test_input(training_set), dtype=float)
    # print('Expected      Result')

    # for i in range(results.size):
    #     r=1
    #     if(results[i][0]<0):
    #         r=-1
    #     print(f'{expected_output[i][0]}\t\t{r}')
