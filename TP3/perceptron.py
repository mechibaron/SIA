import numpy as np

def step_function(value):
    if value < 0:
        return -1
    else:  
        return 1


def predict(data, w):
    #print(data)
    #print(w[1:])
    value = np.dot(data, w[1:]) - w[0]
    print("Predict value: ", value)
    return step_function(value)

def update_weigths(w,learning_rate,sourceData,error, bias):
    for i in range(1,len(w)):
        w[i] = w[i] + learning_rate * error * sourceData[i-1]

    # w[1:] += [learning_rate * error * float(i) for i in sourceData]
    w[0] += learning_rate * error * bias
    return w

def converged(expectedData, output):
    return False