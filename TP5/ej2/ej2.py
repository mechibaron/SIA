from tensorflow import keras
import numpy as np
from src.variational_autoencoder import VariationalAutoencoder
from src.plotting import plotAverages, plotLatent

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
trainset = np.concatenate([x_train, x_test], axis=0)
trainset = np.expand_dims(trainset, -1).astype("float32") / 255
vae = VariationalAutoencoder()
plotLatent(vae)
vae.train(trainset)
plotLatent(vae)
trainset = np.expand_dims(x_train, -1).astype("float32") / 255
trainoutputset = np.concatenate([y_train, y_test]).astype("float32") / 255
trainoutputset = np.expand_dims(trainoutputset, -1).astype("float32") / 255
print("VAE ", vae)
print("TRAINSET ", trainset)
print("Y_TRAIN ", y_train)
plotAverages(vae, trainset, y_train)