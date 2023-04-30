import matplotlib.pyplot as plt
import numpy as np
def punto_corte(x0, y0, m):
    m_ortogonal =  -1/m
    b_ortogonal = y0 - m_ortogonal * x0
    return b_ortogonal

def plot(w, operation, data, result, epoch):

    x_coords = [point[0] for point in data]
    y_coords = [point[1] for point in data]

    colors = ['blue' if point == -1 else 'red' for point in result]
    # Crear el gráfico
    fig, ax = plt.subplots()
    for i, point in enumerate(data):
        ax.plot(x_coords[i], y_coords[i], 'o', color=colors[i]) # Pintar los puntos según su correspondencia
    x_vals = w[1]
    y_vals = w[2]
    m = -x_vals/y_vals
    x = range(-10,10)
    y= [m*xi - w[0] for xi in x]
    # y = [(m/b)*xi - (m/b)*(-c/b) for xi in x]
    # y_o = [(m_o/b)*xi - (m_o/b)*(-c/b) for xi in x]
    plt.ylim(-1.5,1.5)
    plt.xlim(-1.5,1.5)
    plt.plot(x,y)
    # plt.plot(x,y_o)
    #w[1]*x1 + w[2]*x2 - w[0]
    #ax.plot(x_vals, y_vals, '-', color='green', label='Weight') # '--' indica que se plotee la línea como una línea punteada
    # Agregar títulos y etiquetas a los ejes
    ax.set_xlabel('Coordenada X')
    ax.set_ylabel('Coordenada Y')
    title = 'Puntos de la función get_data() para la operacion ' + operation + '\nen la epoca ' + str(epoch)
    ax.set_title(str(title))
    # ax.set_label("Epoca " + epoch)
    # Mostrar el gráfico
    plt.show()