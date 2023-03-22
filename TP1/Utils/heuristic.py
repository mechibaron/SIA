import Utils.fillzoneUtils as fillzoneUtils

# Cantidad de colores restantes para completar la matriz (no se tiene en cuenta al bloque principal)
def heuristic1(actual_node, dim, colors):
    color_list = []
    for i in range(dim):
        for j in range(dim):
            if actual_node.visited[i][j] == 0 and actual_node.state[i][j] not in color_list:
                color_list.append(actual_node.state[i][j])
            if color_list.__len__() == colors:
                break
        if color_list.__len__() == colors:
            break
    return color_list.__len__()


# Maxima cantidad de bloques adyacentes del mismo color != mi color
def heuristic2(matrix, actual_node, dim):
    dx = [-1, 0, 1, 0]  # cambios en x para obtener los vecinos
    dy = [0, 1, 0, -1]  # cambios en y para obtener los vecinos

    # Me fijo en que posicion [x,y] esta el nodo actual
    flag = True
    found_island = False
    x=0
    y=0
    i=0
    j=0
    while i<dim and flag:
        while j<dim and flag:
            if actual_node.color == matrix[i][j] :
                # Encontre el bloque
                k=i+1
                l=j
                while k<dim and flag:
                    while l<dim and flag:
                        # Si es distinto sali del bloque
                        if(actual_node.color!=matrix[k][l]):
                            if(l==0):
                                x=k-1
                                y=dim
                            else:
                                x=k
                                y=l-1
                            flag=False
                        l+=1
                    l=0
                    k+=1
            j += 1
        j=0
        i += 1


    # Me guardo en color_list los colores de los 4 vecinos tales que sean distintos
    # al color del nodo actual
    color_list = []
    for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]
        if((fillzoneUtils.in_grid(nx, ny, dim)==True)):
            if (matrix[nx][ny] != actual_node.color):
                color_list.append(matrix[nx][ny])

    # Veo la cantidad de veces que se repite cada color en dicha lista
    max_repetitions = 0
    for color in color_list:
        if color_list.count(color)>max_repetitions :
            max_repetitions = color_list.count(color)

    # Tomo el que mas veces se repitio y se divide por 4 que seria por la cantidad de lugares posibles
    return max_repetitions/4


# Cantidad de bloques restantes
def heuristic3(actual_node, dim):
    return dim * dim - actual_node.island_size
