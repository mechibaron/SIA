from tensorflow import keras

from ..multilayer.variational_autoencoder import VariationalAutoencoder
from src.plotting import *
# import certifi
from src.plotting import *
# from keras.datasets import imdb
# from keras.datasets import reuters
import requests
requests.packages.urllib3.disable_warnings()
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

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