import step
import non_linear
import linear
import json
import utils
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

testing_size = [0.1, 0.2, 0.3, 0.4, 0.5]
random_values = [10,35,42]
learning_rate_variation = [0.1, 0.01, 0.001, 0.0001]
beta_variation = [0.3, 0.8, 1]


def init():
    X = []
    Z = []
    with open('./data/TP3-ej2-conjunto.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            X.append([float(row['x1']), float(row['x2']), float(row['x3'])])
            Z.append(float(row['y']))
    X = np.array(X)
    Z = np.array(Z)
    return X,Z

def test_seed_and_size(learning_rate, epochs, beta, bias,theta):
    X, Z = init()

    df = {}

    for random in random_values:
        key = 'Random state: ' + str(random)
        df[key] = {}
        for test in testing_size:
            df[key][test] = []
            X_train, X_test, Z_train, Z_test = train_test_split(X,Z,test_size=test,train_size=1-test, random_state=random)

            # perc = non_linear.NonlinearPerceptron(learning_rate, epochs, len(X[0]), theta, beta, bias)
            perc = linear.LinearPerceptron(learning_rate,len(X[0]),epochs,bias)
            perc.train_online(X_train, Z_train)

            df[key][test] = perc.test(X_test, Z_test)[0]


    df_final = pd.DataFrame(df)   
    print(df_final)
    ax = df_final.plot.bar()
    ax.set_title("Variation on MSE for Linear Perceptron")
    ax.set_xlabel("Test size")
    ax.set_ylabel("Mean Square Error")
    # ax.set_ticks(["Random-state: 10","Random-state: 35","Random-state: 42"])
    ax.set_xticklabels(["10%", "20%", "30%", "40%", "50%"], rotation=50)
    plt.show()

def show_eror_on_epocs(learning_rate, epochs, beta, bias):
    X, Z = init()
    
    X_train, X_test, Z_train, Z_test = train_test_split(X,Z,test_size=0.2, random_state=10)

    perc = linear.LinearPerceptron(learning_rate, len(X[0]), epochs, bias)
    error_linear = perc.train_online(X_train, Z_train)

    perc = non_linear.NonlinearPerceptron(learning_rate, epochs, len(X[0]), theta, beta, bias)
    error_nonlinear = perc.train_online(X_train, Z_train)

    df = pd.DataFrame({"Linear":error_linear, "Nonlinear": error_nonlinear})
    print(df)

    # plt.plot(df)
    plt.plot(df['Linear'], label='Linear')
    plt.plot(df['Nonlinear'], label='Nonlinear')    
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title("Error variation in between epochs")
    plt.show()

def show_mse_with_ETA(epochs, beta, bias, theta):
    X, Z = init()

    X_train, X_test, Z_train, Z_test = train_test_split(X,Z,test_size=0.2, random_state=10)

    dict = {}
    dict['Linear'] = []

    errors = {}
    errors['Linear'] = {}

    w_linear = [[0.59221049, 0.47456224, 0.36653657, 0.44478228],
                [0.71793471, 0.43590919, 0.68270929, 0.96992507],
                [0.14636744, 0.92831909, 0.84248777, 0.92049968],
                [0.37725286, 0.56415224, 0.43206012, 0.31905116]]

    w_nonlinear = [[0.75986395,0.06073236,0.90783762,0.70560223],
         [0.4506844,0.2049456, 0.1856947, 0.0925782],
         [0.28030162, 0.81770428, 0.25112686, 0.65157396],
         [0.55724753, 0.00147408, 0.62403554, 0.38887813]]

    i=0
    for learning_rate in learning_rate_variation:
        perc = linear.LinearPerceptron(learning_rate, len(X[0]), epochs, bias,w_linear[i])
        errors['Linear'][learning_rate] = (perc.train_online(X_train, Z_train))
        dict['Linear'].append(perc.test(X_test, Z_test)[0])
        i+=1

    dict['Nonlinear'] = []
    errors['Nonlinear'] = {}

    i=0
    for learning_rate in learning_rate_variation:
        perc = non_linear.NonlinearPerceptron(learning_rate, epochs, len(X[0]), theta, beta, bias,w_nonlinear[i])
        errors['Nonlinear'][learning_rate] = (perc.train_online(X_train, Z_train))
        dict['Nonlinear'].append(perc.test(X_test, Z_test)[0])
        i+=1


    df = pd.DataFrame(dict)
    
    # print(pd.DataFrame(errors['Nonlinear']))
    # df = pd.DataFrame(errors['Nonlinear'])
    # ax = df.plot.bar()
    # ax.legend()
    # ax.set_xlabel('Learning Rates')
    # ax.set_ylabel('MSE')
    # ax.set_title("Error for different learning rates")
    # ax.set_xticklabels(learning_rate_variation, rotation=45)
    # plt.show()

    print(pd.DataFrame(errors['Linear']))
    df = pd.DataFrame(errors['Linear'])
    df.index = range(1, len(df) + 1)
    df.plot(style='o')
    plt.title("Error variation on epochs for Linear Perceptron")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.ylim(0,200)
    plt.xlim(0,50)
    plt.show()
    # ax = df.plot()
    # ax.set_xlabel("Epochs")
    # ax.set_ylabel("MSE")
    # ax.set_title("Error variation on epochs for Nonlinear Perceptron")
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
    # ax.legend(loc='upper right',bbox_to_anchor=(1, 1))
    plt.show()

def show_variation_beta(learning_rate, epochs, bias, theta):
    X, Z = init()

    X_train, X_test, Z_train, Z_test = train_test_split(X,Z,test_size=0.2, random_state=10)

    w = [[0.07390637, 0.21945717, 0.39816684, 0.4646161 ],
[0.1692354,  0.06999783, 0.66982568, 0.37643645],
[0.31565424, 0.40489565, 0.76876674, 0.67414795]]

    errors = {}
    # errors['Nonlinear'] = {} 
    i = 0
    for beta in beta_variation:
        perc = non_linear.NonlinearPerceptron(learning_rate, epochs, len(X[0]), theta, beta, bias,w[i])
        errors["Beta: " + str(beta)] = perc.train_online(X_train, Z_train)
        i+=1
        # errors['Nonlinear'].append(perc.test(X_test, Z_test)[0])

    df = pd.DataFrame(errors)
    df.index = range(1, len(df) + 1)

    print(df)
    ax = df.plot(style='o')
    ax.set_title("Beta variations for tanh function for nonlinear")
    ax.set_xlabel("Epocs")
    ax.set_ylabel("MSE")
    # ax.set_xlim(0,50)
    plt.show()

def show_error_by_epoch(learning_rate, epochs, bias, beta, theta):
    X, Z = init()
    
    X_train, X_test, Z_train, Z_test = train_test_split(X,Z,test_size=0.2, random_state=42)
    # perc = non_linear.NonlinearPerceptron(learning_rate, epochs, len(X[0]), theta, beta, bias)
    # error = perc.train_online(X_train, Z_train)
    perc = linear.LinearPerceptron(learning_rate,len(X[0]), epochs, bias)
    error = perc.train_online(X_train, Z_train)

    df = pd.DataFrame(error)

    print(df)
    df.index = range(1, len(df) + 1)
    # Filter out null values
    plt.scatter(df.index, df[0])
    plt.xlim(0,50)
    plt.ylim(0,200)
    # plt.xticks([i for i in range(100)])
    plt.title("MSE vs Epochs for Linear Perceptron")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    # plt.xticks(range(1, 100))
    plt.show()

if __name__ == '__main__':
    with open('./json/config.json', 'r') as f:
        data = json.load(f)
        f.close()
  # define learning rate and epochs
  # learning rate = eta (n)

    learning_rate, epochs, bias, type_perceptron = utils.getDataFromFile(data)
    beta, theta = 0,0
    if (type_perceptron != 'lineal'):
        with open('./json/ej2_config.json', 'r') as f:
            ej2 = json.load(f)
            f.close()
        beta, theta = utils.getDataForEj2(ej2)
    show_variation_beta(learning_rate,epochs,bias, theta)
    # operation, learning_rate, epochs, bias, beta, type_perceptron, theta = utils.getDataFromFile(data)
    # test_seed_and_size(learning_rate,epochs,beta, bias, theta)
    # show_error_by_epoch(learning_rate,epochs,beta, bias, theta)
    # show_mse_with_ETA(epochs,beta, bias, theta)
 