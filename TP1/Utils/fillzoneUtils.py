import numpy as np
import node

# index = 0 ---> ARRIBA
# index = 1 ---> IZQUIERDA
# index = 2 ---> DERECHA
# index = 3 ---> ABAJO
row = [-1, 0, 0, 1]
col = [0, -1, 1, 0]
movement_cost = 1

def in_grid(x, y, dim):
    if (x >= 0 and x < dim and y >= 0 and y < dim):
        return True
    return False

# devuelve una matriz con 1's donde se ubica nuestra isla principal
def get_principal_block_recursive(matrix, visited, i, j, color, island_size, dim):
    if i < 0 or j < 0 or i >= dim or j >= dim or visited[i][j]:
        return visited, island_size

    if matrix[i][j] == color:
        visited[i][j] = True
        island_size += 1
        new_visited = visited

        for k in range(4):
            new_visited, island_size = get_principal_block_recursive(matrix, new_visited, i + row[k], j + col[k],
                                                           color, island_size, dim)
        return new_visited, island_size

    return visited, island_size


# cambia el color de la isla principal
def change_color(state, visited, color, dim):
    matrix = np.copy(state)
    for i in range(dim):
        for j in range(dim):
            if visited[i][j]:
                matrix[i][j] = color
    return matrix


def is_goal(actual_node, dim):
    for i in range(dim):
        for j in range(dim):
            if actual_node.state[i][j] != actual_node.color:
                return False
    return True


def is_insignificant_move(visited, new_visited, dim):
    for i in range(dim):
        for j in range(dim):
            if visited[i][j] != new_visited[i][j]:
                return False
    return True


def get_best_color(actual_node, perimetral_colors, numeral, dim):
    best_node = actual_node
    if len(perimetral_colors) == 0:
        return 0, None

    for perimetral_color in perimetral_colors:
        new_state = change_color(actual_node.state, actual_node.visited, perimetral_color, dim)
        blank_matrix = np.zeros((dim, dim))
        main_island, island_size = get_principal_block_recursive(new_state, blank_matrix, 0, 0, perimetral_color, 0, dim)
        if best_node.island_size < island_size:
            new_node = node.Node(new_state, main_island, actual_node.cost + movement_cost,
                                 actual_node, perimetral_color, island_size)
            best_node = new_node
    # no cambio el tamaño por lo que no tiene vecinos o algo anda mal
    if best_node.island_size == actual_node.island_size:
        return -1, None

    perimetral_colors.remove(best_node.color)

    return numeral / best_node.island_size, best_node