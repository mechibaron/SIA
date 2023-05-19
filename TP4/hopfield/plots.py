import matplotlib.pyplot as plt

def plot_letter(letter):
    # Crear una figura y un conjunto de ejes
    fig, ax = plt.subplots()

    # Pintar la matriz en un gráfico de colores
    ax.imshow(letter, cmap='Blues')

    # Mostrar el gráfico
    plt.show()