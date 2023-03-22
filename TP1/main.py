import numpy as np
import node
import time
import Utils.fillzoneUtils as fillzoneUtils
import Utils.priorityQueue as priorityQueue
import Utils.priorityQueueGreedy as priorityQueueGreedy
import Utils.heuristic as heuristics
movement_cost = 1

colors = int(input('Ingrese número de colores (M): '))
dim = int(input('Ingrese dimensión del tablero (N): '))
print('Métodos de búsqueda')
print('1 - DFS')
print('2 - BFS')
print('3 - A*')
print('4 - Greedy')
search_method = int(input('Seleccione un método de búsqueda: '))
if search_method == 4 or search_method == 3:
    print('Heurísticas')
    print('1 - Cantidad de colores restantes')
    print('2 - Cantidad de bloques adyacentes de igual color distinto a si mismo')
    print('3 - Bloques restantes')
    heuristic = int(input('Seleccione una heurística: '))


# armo una cola y voy sacando el nodo que hace más tiempo se encuentra en la cola
def bfs_search_fill_zone(root):
    queue = [root]
    total_nodes = 1
    border_nodes = 1

    while queue:
        actual_node = queue.pop(0)
        border_nodes = border_nodes - 1

        if fillzoneUtils.is_goal(actual_node, dim):
            return actual_node, border_nodes, total_nodes

        # por cada color veo como queda la matriz al escogerlo
        for color in range(colors):
            if color != actual_node.color:
                new_state = fillzoneUtils.change_color(np.copy(actual_node.state), actual_node.visited, color, dim)
                blank_matrix = np.zeros((dim, dim))
                main_island, island_size = fillzoneUtils.get_principal_block_recursive(new_state, blank_matrix, 0, 0, color, 0, dim)
                if not fillzoneUtils.is_insignificant_move(actual_node.visited, main_island, dim):
                    new_node = node.Node(new_state, main_island, actual_node.cost + movement_cost,
                                         actual_node, color, island_size)
                    queue.append(new_node)
                    total_nodes = total_nodes + 1
                    border_nodes = border_nodes + 1


def dfs_search_fill_zone(actual_node, border_nodes_dfs=1, total_nodes_dfs=1):
    border_nodes_dfs = border_nodes_dfs - 1
    # nodo Encontrado
    if fillzoneUtils.is_goal(actual_node, dim):
        return actual_node, border_nodes_dfs, total_nodes_dfs

    # siguiente nodo
    for color in range(colors):
        if color != actual_node.color:
            new_state = fillzoneUtils.change_color(np.copy(actual_node.state), actual_node.visited, color, dim)
            blank_matrix = np.zeros((dim, dim))
            main_island, island_size = fillzoneUtils.get_principal_block_recursive(new_state, blank_matrix, 0, 0, color, 0, dim)
            if not fillzoneUtils.is_insignificant_move(actual_node.visited, main_island, dim):
                new_node = node.Node(new_state, main_island, actual_node.cost + movement_cost,
                                     actual_node, color, island_size)
                next_node, border_nodes, total_nodes = dfs_search_fill_zone(new_node, border_nodes_dfs + 1,
                                                                  total_nodes_dfs + 1)
                if next_node is not None:
                    return next_node, border_nodes, total_nodes



def a_search_fill_zone(root):
    queue = priorityQueue.PriorityQueue()
    queue.insert(root)

    total_nodes = 1
    border_nodes = 1
    gained_blocks = 0
    while not queue.isEmpty():
        actual_node = queue.pop()
        border_nodes = border_nodes - 1
        if fillzoneUtils.is_goal(actual_node, dim):
            return actual_node, border_nodes, total_nodes

        # Por cada color veo como queda la matriz al escogerlo
        for color in range(colors):
            if color != actual_node.color:
                new_state = fillzoneUtils.change_color(np.copy(actual_node.state), actual_node.visited, color, dim)
                gained_blocks += 1
                blank_matrix = np.zeros((dim, dim))
                main_island, island_size = fillzoneUtils.get_principal_block_recursive(new_state, blank_matrix, 0, 0, color, 0, dim)
                if actual_node.island_size < island_size:
                    new_node = node.Node(new_state, main_island, actual_node.cost + movement_cost,
                                         actual_node, color, island_size)
                    if heuristic == 1:
                        heuristic_val = heuristics.heuristic1(new_node, dim, colors)
                    elif heuristic == 2:
                      heuristic_val = heuristics.heuristic2(new_state, new_node, dim)
                    else:
                       heuristic_val = heuristics.heuristic3(new_node, dim)

                    new_node.set_value(heuristic_val)

                    if fillzoneUtils.is_goal(new_node, dim):
                        return actual_node, border_nodes, total_nodes

                    queue.insert(new_node)
                    total_nodes = total_nodes + 1
                    border_nodes = border_nodes + 1


def greedy_fill_zone(root):
    current = root
    total_nodes = 1
    border_nodes = 0
    gained_blocks = 0
    while not fillzoneUtils.is_goal(current, dim):
        queue = priorityQueueGreedy.PriorityQueue()
        # Por cada color veo como queda la matriz al escogerlo
        for color in range(colors):
            if color != current.color:
                aux = np.copy(current.state)
                new_state = fillzoneUtils.change_color(aux, current.visited, color, dim)
                gained_blocks += 1
                blank_matrix = np.zeros((dim, dim))
                main_island, island_size = fillzoneUtils.get_principal_block_recursive(new_state, blank_matrix, 0, 0, color, 0, dim)
                if not fillzoneUtils.is_insignificant_move(current.visited, main_island, dim):
                    new_node = node.Node(new_state, main_island, current.cost + movement_cost,
                                         current, color, island_size)
                    if heuristic == 1:
                        heuristic_val = heuristics.heuristic1(new_node,dim,color)
                    elif heuristic == 2:
                      heuristic_val = heuristics.heuristic2(new_state, new_node, dim)
                    else:
                        heuristic_val = heuristics.heuristic3(new_node, dim)  

                    new_node.set_value(heuristic_val)
                    queue.insert(new_node)
                    total_nodes = total_nodes + 1
                    border_nodes = border_nodes + 1
        current = queue.pop()
        border_nodes = border_nodes - 1

    return current, border_nodes, total_nodes


def main():

    random_matrix = np.random.randint(0, colors, (dim, dim))

    visited = np.zeros((dim, dim))

    main_island, island_size = fillzoneUtils.get_principal_block_recursive(random_matrix, visited, 0, 0, random_matrix[0][0], 0, dim)

    root = node.Node(random_matrix, main_island, 0, None,
                     random_matrix[0][0], island_size)

    start = time.time()
    if search_method == 2:
        goal, border_nodes, total_nodes = bfs_search_fill_zone(root)
    elif search_method == 3:
        goal, border_nodes, total_nodes = a_search_fill_zone(root)
    elif search_method == 4:
        goal, border_nodes, total_nodes = greedy_fill_zone(root)
    else:
        goal, border_nodes, total_nodes = dfs_search_fill_zone(root)

    end = time.time()

    current = goal
    while current is not None:
        print('Color elegido: ' + str(current.color))
        print(current.state)
        print('                 ')
        current = current.parent

    print('Costo total: ' + str(goal.cost))
    print('Nodos expandidos: ' + str(total_nodes))
    print('Nodos frontera: ' + str(border_nodes))
    print('Tiempo de procesamiento: ' + str(end - start) + ' seg')
    


if __name__ == "__main__":
    main()

