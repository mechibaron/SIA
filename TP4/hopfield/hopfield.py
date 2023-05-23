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
        self.weights = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if(i>j):
                    w_ij = self.sinaptic_weights(i,j)
                    self.weights[i][j] = w_ij
                    self.weights[j][i] = w_ij
        self.state_energy = []

    def sinaptic_weights(self, i, j):
        summatory = 0
        for mu in range(len(self.training_letters)):
            summatory+=self.training_letters[mu][i] * self.training_letters[mu][j] 
        return summatory/self.n

    def train(self,noise_letter):
        # Inicializo el estado con el patron de consulta (noise_letter)
        state_neurons = noise_letter.flatten()
        self.state_energy.append(self.energy(state_neurons))
        # print("weights: \n", self.weights)
        for e in range(self.epochs):
            print("Iteration: ", e)
            # Si no converge continuo con hopfield
            # print(state_neurons)
            converge, idx = self.converge(state_neurons)
            previous_state = state_neurons
            state_neurons = self.hopfield(state_neurons)
            response = np.array(state_neurons).reshape((self.mu, self.mu))

            if(np.array_equal(previous_state, state_neurons) == True):
                if (converge == True):
                    response =  self.matrix_training_letters[idx]
                    break
                else: 
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

    def hopfield(self, state_neurons):
        new_state = []

        for i in range(self.n):
            new_state.append(self.step_function(np.inner(self.weights[i], state_neurons)))
        self.state_energy.append(self.energy(new_state))
        return new_state
        
            
    def converge(self, state_neurons):
        for l in range(len(self.training_letters)):
            if (np.array_equal(self.training_letters[l], state_neurons) == True):
                print("Found equal in training = ", l)
                return True, l
        return False, -1

    def energy(self, state_neurons):
        result = 0
        for i in range(self.n):
            for j in range(self.n):
                result += self.weights[i][j] * state_neurons[i] * state_neurons[j]
        return -0.5 * result
    
    def step_function(self,h):
        if h < 0:
            return -1
            
        elif h > 0:  
            return 1
        else:
            return h