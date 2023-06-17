from tensorflow import keras

from src.variational_autoencoder import VariationalAutoencoder
from src.plotting import *

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
trainset = np.concatenate([x_train, x_test], axis=0)
trainset = np.expand_dims(trainset, -1).astype("float32") / 255
vae = VariationalAutoencoder()
vae.train(trainset)
plotLatent(vae)
trainset = np.expand_dims(x_train, -1).astype("float32") / 255
trainoutputset = np.concatenate([y_train, y_test]).astype("float32") / 255
trainoutputset = np.expand_dims(trainoutputset, -1).astype("float32") / 255
plotAverages(vae, trainset, y_train)