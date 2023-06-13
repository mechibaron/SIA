import numpy as np

class Autoencoder:
    def __init__(self,input_size, latent_size):
        self.input_size = input_size # matriz de nxd 
        self.latent_size = latent_size # matriz de nxk
        self.weights = np.array((len(input_size[0]),len(latent_size))) # matriz de kxd (asociada al decoder)
        self.encoder = None #multilayer? (misma cantidad de hidden layers)
        self.decoder = None #multilayer?

        self.build()

    def build():
        self.build_encoder()
        self.build_dencoder()
        self.build_autoencoder()

    def build_encoder():
        pass
    
    def build_dencoder():
        pass

    def build_autoencoder():
        pass