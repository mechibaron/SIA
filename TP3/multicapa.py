import numpy as np
import initdata


class MultiCapaPerceptron:
    def __init__(self, learning_rate, input_size, epochs, hidden_layers, amount_of_each_hidden, X,  theta_method, item = "a", bias=1 ):
        self.input_size = input_size
        self.bias_num=bias
        self.bias = np.empty(shape=hidden_layers+1, dtype=object)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.item = item
        self.hidden_layers = hidden_layers
        self.amount_of_each_hidden = amount_of_each_hidden
        self.W = np.empty(hidden_layers+1, dtype=object)
        self.W0= np.empty(hidden_layers+1, dtype=object)
        self.V= np.empty(hidden_layers+1, dtype=object)
        self.theta_method=theta_method
        self.prev_V=X
        self.X=X
        self.error=0
    
    def tanh(self, x):
        return np.tanh(self.beta * x)

    def d_tanh(self, x):
        return self.beta * (1 - (self.tanh(x)) ** 2)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-2 * self.beta *x))

    def d_sigmoid(self, x):
        return 2*self.beta*self.sigmoid(x)*(1-self.sigmoid(x))
    
    def step_function(self,x):
        x = np.where(x < 0, -1, 1)
        return x
    
    def initialize(self):
        for i in range(self.hidden_layers + 1):
            prev = self.amount_of_each_hidden[i-1]
            if(i==0):
                prev = self.input_size
                print("i=0 : ", prev , self.amount_of_each_hidden[i])
            if(i==self.hidden_layers):
                col = 1
            else:
                col = self.amount_of_each_hidden[i]

            self.W[i] = np.random.rand(prev, col)*0.01
            self.W0[i] = np.random.rand(len(self.X),col)*0.01
            self.bias[i] = np.full((len(self.X), col), self.bias_num)
        
    def predict(self, index):
        # 4x2 * 2x.
        #[[x11, x12],[x21,x22],[x31, x32],[x41, x42]]*[w1,w2,w3,w4] + [w0_1, w0_2, w0_3, w0_4] 
        # print(self.prev_V)
        # print(self.W[index])
        mult = np.dot(self.prev_V,self.W[index]) + self.W0[index]
        # print(mult)
        theta = self.theta_method[index]
        if(theta=="lineal"):
            return mult
        elif (theta=="no_lineal"):
            if(theta[1] =="tanh"):
                return self.tanh(mult)
            else:
                return self.sigmoid(mult)
        else:
            return self.step_function(mult)
        
    def forward_propagation(self,index):
        self.V[index] = self.predict(index)
        self.prev_V = self.V[index] 

    def backward_propagation(self,index, weigths):
        delta=self.error
        print("delta:\n", delta)
        print("V:\n", self.V[index])
        print("w:\n", weigths[index])
        print("eta:\n", self.learning_rate)
        theta = self.theta_method[index]
        if (theta=="no_lineal"):
            h = np.dot(self.V[index], weigths[index])
            if(theta[1] =="tanh"):
                delta = delta*self.d_tanh(h)
            else:
                delta = delta*self.d_sigmoid(h)
        return self.learning_rate*np.dot(delta,self.V[index])
    
    def backward_propagation_w0(self,index, weigths):
        delta=self.error
        print("delta:\n", delta)
        print("bias:\n", self.bias[index])
        print("w:\n", weigths[index])
        print("eta:\n", self.learning_rate)
        theta = self.theta_method[index]
        if (theta=="no_lineal"):
            h =  weigths[index]
            # h = np.dot(self.V[index], weigths[index])
            if(theta[1] =="tanh"):
                delta = delta*self.d_tanh(h)
            else:
                delta = delta*self.d_sigmoid(h)
        return self.learning_rate*np.dot(delta,self.bias[index])
    #0.1*(1x4 , 4x3) => 1x3
    
    def train_online(self, Z):
        # print("EPOCAS", self.epochs)
        for i in range(self.epochs):
            print(self.epochs)
            for j in range(self.hidden_layers + 1):
                self.forward_propagation(j) #Cuando termina de ciclar queda guardado Output en prev_V
            self.V[self.hidden_layers]=self.V[self.hidden_layers]
            self.prev_V = self.prev_V.flatten()
            self.error=((-1)*(Z - self.prev_V))
            print(self.W)
            for j in range(self.hidden_layers, -1, -1):
                print("\nJ: ", j)
                self.W[j] += self.backward_propagation(j, self.W)
                print("SELF W0 :",self.W0)
                bp =  self.backward_propagation_w0(j, self.W0)
                # delta= 1x4, bias = 4x3 => bp = 1x3
                print("BP: ",bp)
                self.W0[j] += bp
              
def main(learning_rate, epochs, bias, item):
    hidden_layers = 0
    if(item =="a"):
        X = initdata.get_data()
        Z = initdata.get_data_expected("XOR")  
        hidden_layers = 3
        amount_of_each_hidden=[3,2,3]
        theta_method=[["lineal"],["no_lineal", "tanh"],["step"], ["step"]] #longitud = hidden layers + 1 (output)
        
    perc = MultiCapaPerceptron(learning_rate, len(X[0]), epochs, hidden_layers, amount_of_each_hidden,X,theta_method, item, bias)
    perc.initialize()
    perc.train_online(Z)


if __name__ == '__main__':
    main()
