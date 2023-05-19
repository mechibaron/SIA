import numpy as np
import matplotlib.pyplot as plt
from hopfield import plots

class Hopfield:

    def __init__(self, epochs, mu, training_letters):
        self.epochs = epochs
        self.mu = mu
        self.n = mu**2
        self.matrix_training_letters = training_letters
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
            print("Iteration: ", e)
            # Si no converge continuo con hopfield
            converge, idx = self.converge()
            previous_state = self.state_neurons
            self.hopfield()
            response = self.state_neurons.reshape((self.mu, self.mu))
            # Lo que nos esta pasando es que es estado se actuliza a lo mas cercano a la primera iteracion y no avanza desde ahi
            # if(converge == True):
            #     if (np.array_equal(previous_state, self.state_neurons) == True):
            #         response =  self.matrix_training_letters[idx]
            #         break
            #     else: 
            #         response = self.state_neurons.reshape((self.mu, self.mu))
            #         break
            if(np.array_equal(previous_state, self.state_neurons) == True):
                if (converge == True):
                    response =  self.matrix_training_letters[idx]
                    break
                else: 
                    response = self.state_neurons.reshape((self.mu, self.mu))
                    break

        
        new_list = range(0, e+1)
        plt.title("Energy levels per epoch")
        plt.plot([i for i in range(len(self.state_energy))], self.state_energy)
        plt.ylabel('Energy level')
        plt.xlabel('Epochs')
        plt.xticks(new_list)
        plt.show()

        # Si no encontre estado estable devuelvo false y el ultimo estado alcanzado
        return response

    def hopfield(self):
        new_state = []
        for i in range(self.n):
            new_state = self.step_function(np.inner(self.weights[i], self.state_neurons))
            self.state_neurons[i] = new_state
        self.state_energy.append(self.energy())
        
            
    def converge(self):
        for l in range(len(self.training_letters)):
            if (np.array_equal(self.training_letters[l], self.state_neurons) == True):
                return True, l
        return False, -1

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