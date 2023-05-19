import numpy as np
import matplotlib.pyplot as plt

class Hopfield:

    def __init__(self, epochs, mu, training_letters):
        self.epochs = epochs
        self.mu = mu
        self.n = mu**2
        self.training_letters = [matriz.flatten() for matriz in training_letters]
        self.p = len(self.training_letters)
        # Weights transpose
        self.weights = np.dot(np.transpose(self.training_letters), self.training_letters) / self.n
        np.fill_diagonal(self.weights, 0)

        self.state_neurons = []
        self.state_energy = []

    def train(self,noise_letter):
        # Inicializo el estado con el patron de consulta (noise_letter)
        self.state_neurons = noise_letter.flatten()
        self.state_energy.append(self.energy())
        for e in range(self.epochs):
            # Si no converge continuo con hopfield
            print(e)
            if(self.converge() == False):
                self.hopfield()
                self.state_energy.append(self.energy())
                print("state:",self.state_neurons)
            else:
                f_s = self.state_neurons
                self.hopfield()
                self.state_energy.append(self.energy())
                # Si es estados se mantiene => estado estable
                if(np.array_equal(f_s, self.state_neurons) == True):
                    # Devuelvo el mas similar a noise_letter de training_letters
                    return True, self.find_in_matrix()

        
        new_list = range(0, e+1)
        plt.title("Energy levels per epoch")
        plt.plot([i for i in range(len(self.state_energy))], self.state_energy)
        plt.ylabel('Energy level')
        plt.xlabel('Epochs')
        plt.xticks(new_list)
        plt.show()
        # Si no encontre estado estable devuelvo false y el ultimo estado alcanzado
        return False, self.state_neurons.reshape((self.mu, self.mu))

    def hopfield(self):
        for i in range(self.n):
            new_state = np.inner(self.weights[i], self.state_neurons)
            self.state_neurons[i] = self.step_function(new_state)

    def find_in_matrix(self):
        for i, fila in enumerate(self.training_letters):
            if np.all(fila == self.state_neurons):
                 return i
            
    def converge(self):
        for l in range(len(self.training_letters)):
            if (np.array_equal(self.training_letters[l], self.state_neurons) == True):
                return True
        return False

    def energy(self):
        result = 0
        for i in range(self.n):
            for j in range(self.n):
                result += self.weights[i][j] * self.state_neurons[i] * self.state_neurons[j]
        return -0.5 * result
    
    def step_function(self,h):
        if h < 0:
            return -1
        elif h > 0:  
            return 1
        else:
            return h