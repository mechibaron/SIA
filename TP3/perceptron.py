import numpy as np

def step_function(value):
    if value < 0:
        return -1
    else:  
        return 1


def predict(data, w):
    value = np.dot(data, w[1:]) + w[0]
    return step_function(value)

def update_weigths(w,learning_rate,sourceData,error):
    w[1:] += [learning_rate * error * float(i) for i in sourceData]
    w[0] += learning_rate * error
    return w

def converged(expectedData, output):
    return False