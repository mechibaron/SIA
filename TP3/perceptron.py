import numpy as np

def step_function(x):
    if x < 0:
        return -1
    else:  
        return 1
    
def tanh(x, beta):
    return np.tanh(beta * x)

def d_tanh(x,beta):
    return beta * (1 - (np.tanh(beta * x)) ** 2)

def sigmoid(x, beta):
    return 1 / (1 + np.exp(-2 * beta * x))

def d_sigmoid(x,beta):
    return 2*beta*sigmoid(x,beta)*(1-sigmoid(x,beta))

def mean_square_error(z, O):
    return (z-O)**2

def scaled_lineal(O, min_out, max_out):
    return ((O-min_out)/(max_out - min_out))

def predict_linear(data, w):
    return np.dot(data, w[1:]) - w[0]
    # return np.dot(data, w[1:])

def predict(data, w):
    value = np.dot(data, w[1:]) - w[0]
    print("Predict value: ", value)
    return step_function(value)

def update_weigths_linear(w,learning_rate,X,error, bias):
    for j in range(len(X)):    
        for i in range(1,len(w)):
            w[i] = w[i] + learning_rate * np.sum(error[j] * X[j][i-1])

    w[0] += learning_rate * error[0] * bias
    return w

def update_weigths(w,learning_rate,sourceData,error, bias):
    for i in range(1,len(w)):
        w[i] = w[i] + learning_rate * error * sourceData[i-1]

    w[0] += learning_rate * error * bias
    return w

def update_weigths_no_linear(w, learning_rate, error,X, bias, type, beta):
    for j in range(len(X)):    
        for i in range(1,len(w)):
            if(type=="tanh"):
                w[i] = w[i] + learning_rate * np.sum(error[j] * X[j][i-1]*d_tanh(X[j], beta))
            else:
                w[i] = w[i] + learning_rate * np.sum(error[j] * X[j][i-1]*d_sigmoid(X[j], beta))

    w[0] += learning_rate * error[0] * bias
    return w