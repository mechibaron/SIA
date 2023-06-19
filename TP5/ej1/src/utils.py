import numpy as np
import matplotlib.pyplot as plt
from res.fonts import *
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def print_letter(letter):
    test = np.array_split(letter, 5)
    aux = len(test)
    for line in range(0, aux):
        str = ''
        for i in range(0, len(test[0])):
            if test[line][i] > 0.5:
                str = str + '*'
            else:
                str = str + ' '
        print(str)

def graph_digits(original, output):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Original')
    ax2.set_title('AE result')
    ax1.imshow(np.array(original).reshape((7, 5)), cmap='gray')
    ax2.imshow(np.array(output).reshape((7, 5)), cmap='gray')
    fig.show()

def transform(t):
    to_ret = []
    for i in t:
        aux = []
        for num in i:
            a = format(num, "b").zfill(5)
            for j in a:
                if j == "0":
                    aux.append(-1)
                elif j == "1":
                    aux.append(1)
        to_ret.append(aux)
    return np.array(to_ret)

def get_input(font):
    if font == 1:
        return font_1_input
    elif font == 2:
        return font_2_input
    elif font == 3:
        return font_3_input

def get_header(font):
    if font == 1:
        return font_1_header
    elif font == 2:
        return font_2_header
    elif font == 3:
        return font_3_header

def get_output(font):
    if font == 1:
        return font_1_output
    elif font == 2:
        return font_2_output
    else:
        return font_3_output

def noise(t):
    RAND = 1 / 35
    to_ret = []
    for i in t:
        aux = []
        for num in i:
            a = format(num, "b").zfill(5)
            for j in a:
                rand = random.uniform(0, 1)
                if j == "0":
                    if rand < RAND:
                        aux.append(1)
                    else:
                        aux.append(-1)
                elif j == "1":
                    if rand < RAND:
                        aux.append(-1)
                    else:
                        aux.append(1)
        to_ret.append(aux)
    return np.array(to_ret)

def getSample(z_mean, z_log_var):
    z_mean, z_log_var = z_mean, z_log_var
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon
