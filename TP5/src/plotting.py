import numpy as np
import matplotlib.pyplot as plt

def plotAverages(vae, data, labels):
    avg,_,_ = vae.encoder.predict(data)
    colormap = plt.cm.get_cmap('plasma')
    plt.figure(figsize=(12, 10))
    sc = plt.scatter(avg[:, 0], avg[:, 1], c=labels, cmap=colormap)
    plt.colorbar(sc)
    plt.show()


def plotLatent(vae, n=10, figsize=15, digit_size=28):
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = np.linspace(-1.0, 1.0, n)
    grid_y = np.linspace(-1.0, 1.0, n)[::-1]
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z = np.array([[xi, yi]])
            output = vae.decoder.predict(z)
            digit = output[0].reshape(digit_size, digit_size)
            figure[
            i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size,
            ] = digit
    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.imshow(figure, cmap="Greys_r")
    plt.show()