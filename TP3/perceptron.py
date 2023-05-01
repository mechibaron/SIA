import numpy as np

def escale(Z, z, interval):
    max, min = np.max(Z), np.min(Z) 
    if max == min:
        return z * 0
    escaled_data = (interval[1] - interval[0]) * ((z - min) / (max - min)) + interval[0]
    return escaled_data

def escale_all(Z, interval):
    escaled_data = []
    for i in range(len(Z)):
        escaled_data.append(escale(Z,Z[i],interval))
    return escaled_data

def denormalize(Z, O):
    max, min = np.max(Z), np.min(Z) 
    if max == min:
        return O * 0
    return ((O + 1) * (max - min) * 0.5) + min

def step_function(x):
    if x < 0:
        return -1
    else:  
        return 1

def scaled_lineal(O, min_out, max_out):
    return ((O-min_out)/(max_out - min_out))

def predict_linear(data, w):
    return np.dot(data, w[1:]) + w[0]
    # return np.dot(data, w[1:])

def predict(data, w):
    value = np.dot(data, w[1:]) + w[0]
    print("Predict value: ", value)
    return step_function(value)

def update_weigths_linear(w,learning_rate,x,error, bias):
    for i in range(1,len(w)):
        w[i] = w[i] + learning_rate * error * x[i-1]

    w[0] += learning_rate * error * bias
    return w

def update_weigths(w,learning_rate,sourceData,error, bias):
    for i in range(1,len(w)):
        w[i] = w[i] + learning_rate * error * sourceData[i-1]

    w[0] += learning_rate * error * bias
    return w