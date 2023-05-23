import matplotlib.pyplot as plt

def plot_letter(letter, title):
    # Crear una figura y un conjunto de ejes
    fig, ax = plt.subplots()

    # Pintar la matriz en un gráfico de colores
    ax.imshow(letter, cmap='Blues')

    plt.title(title)

    # Mostrar el gráfico
    plt.show()