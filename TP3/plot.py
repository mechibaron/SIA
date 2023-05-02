from non_linear import *
from linear import *
import pandas as pd
import json
import utils
import matplotlib.pyplot as plt 
from scipy import stats


testing_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
random_values = [10,35,42]
learning_rate_variation = [0.1, 0.01, 0.001, 0.0001]

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
            X_train, X_test, Z_train, Z_test = train_test_split(X,Z,test_size=test, random_state=random)

            perc = NonlinearPerceptron(learning_rate, epochs, len(X[0]), theta, beta, bias)
            perc.train_online(X_train, Z_train)

            df[key][test] = perc.test(X_test, Z_test)[0]


    df_final = pd.DataFrame(df)   
    ax = df_final.plot.bar()
    ax.set_title("Variation on MSE for Non Linear Perceptron")
    ax.set_xlabel("Test size")
    ax.set_ylabel("Mean Square Error")
    # ax.set_ticks(["Random-state: 10","Random-state: 35","Random-state: 42"])
    ax.set_xticklabels(["10%", "20%", "30%", "40%", "50%","60%", "70%", "80%", "90%"], rotation=50)
    plt.show()

def show_eror_on_epocs(learning_rate, epochs, beta, bias):
    X, Z = init()
    
    X_train, X_test, Z_train, Z_test = train_test_split(X,Z,test_size=0.2, random_state=10)

    perc = LinearPerceptron(learning_rate, len(X[0]), epochs, bias)
    error_linear = perc.train_online(X_train, Z_train)

    perc = NonlinearPerceptron(learning_rate, epochs, len(X[0]), theta, beta, bias)
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
        perc = LinearPerceptron(learning_rate, len(X[0]), epochs, bias,w_linear[i])
        errors['Linear'][learning_rate] = (perc.train_online(X_train, Z_train))
        dict['Linear'].append(perc.test(X_test, Z_test)[0])
        i+=1

    dict['Nonlinear'] = []
    errors['Nonlinear'] = {}

    i=0
    for learning_rate in learning_rate_variation:
        perc = NonlinearPerceptron(learning_rate, epochs, len(X[0]), theta, beta, bias,w_nonlinear[i])
        errors['Nonlinear'][learning_rate] = (perc.train_online(X_train, Z_train))
        dict['Nonlinear'].append(perc.test(X_test, Z_test)[0])
        i+=1


    # df = pd.DataFrame(dict)
    

    # ax = df.plot.bar()
    # ax.legend()
    # ax.set_xlabel('Learning Rates')
    # ax.set_ylabel('MSE')
    # ax.set_title("Error for different learning rates")
    # ax.set_xticklabels(learning_rate_variation, rotation=45)
    # plt.show()

    print(pd.DataFrame(errors['Nonlinear']))
    df = pd.DataFrame(errors['Nonlinear'])
    ax = df.plot()
    ax.set_xlabel("Epochs")
    ax.set_ylabel("MSE")
    ax.set_title("Error variation on epochs for Linear Perceptron")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
    ax.legend(loc='upper right',bbox_to_anchor=(0.75, 0.75))
    plt.show()

def show_variation_beta(learning_rate, epochs, bias, theta):
    X, Z = init()

    X_train, X_test, Z_train, Z_test = train_test_split(X,Z,test_size=0.2, random_state=10)

    beta_variations= [-1,-0.5, 0, 0.5, 1]

    errors = {}
    errors['Nonlinear'] = [] 
    for beta in beta_variations:
        perc = NonlinearPerceptron(learning_rate, epochs, len(X[0]), theta, beta, bias)
        perc.train_online(X_train, Z_train)
        errors['Nonlinear'].append(perc.test(X_test, Z_test)[0])

    df = pd.DataFrame(errors, index=beta_variations)

    ax = df.plot.bar()
    ax.set_title("Beta variations for tanh function for nonlinear")
    ax.set_xlabel("Beta")
    ax.set_ylabel("MSE")
    ax.legend().remove()
    plt.show()



if __name__ == '__main__':
    with open('./config.json', 'r') as f:
        data = json.load(f)
        f.close()
    operation, learning_rate, epochs, bias, beta, type_perceptron, theta = utils.getDataFromFile(data)
    show_mse_with_ETA(epochs, beta, bias, theta)
 