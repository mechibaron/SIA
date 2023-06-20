import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def getSample(z_mean, z_log_var):
    z_mean, z_log_var = z_mean, z_log_var
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon