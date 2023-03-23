import numpy as np
import Utils.priorityQueue as priorityQueue
import Utils.priorityQueueGreedy as priorityQueueGreedy
import time
import Utils.fillzoneUtils as fillzoneUtils
import Utils.heuristic as heuristics
import node

TRIES = 50

movement_cost = 1

# M
colors = 3
# NxN
# dim = [4, 6, 8]
dim = [4]



# armo una cola y voy sacando el nodo que hace más tiempo se encuentra en la cola
def bfs_search_fill_zone(root, dimension):
    queue = [root]
    total_nodes = 1
    border_nodes = 1

    while queue:
        actual_node = queue.pop(0)
        border_nodes = border_nodes - 1

        if fillzoneUtils.is_goal(actual_node, dimension):
            return actual_node, border_nodes, total_nodes

        # por cada color veo como queda la matriz al escogerlo
        for color in range(colors):
            if color != actual_node.color:
                new_state = fillzoneUtils.change_color(np.copy(actual_node.state), actual_node.visited, color,
                                                       dimension)
                blank_matrix = np.zeros((dimension, dimension))
                main_island, island_size = fillzoneUtils.get_principal_block_recursive(new_state, blank_matrix, 0, 0, color, 0,
                                                                             dimension)
                if not fillzoneUtils.is_insignificant_move(actual_node.visited, main_island, dimension):
                    new_node = node.Node(new_state, main_island, actual_node.cost + movement_cost,
                                         actual_node, color, island_size)
                    queue.append(new_node)
                    total_nodes = total_nodes + 1
                    border_nodes = border_nodes + 1


def dfs_search_fill_zone(actual_node, dimension, border_nodes_dfs=1, total_nodes_dfs=1):
    # nodo Encontrado
    if fillzoneUtils.is_goal(actual_node, dimension):
        return actual_node, border_nodes_dfs, total_nodes_dfs
    # siguiente nodo
    for color in range(colors):
        if color != actual_node.color:
            new_state = fillzoneUtils.change_color(np.copy(actual_node.state), actual_node.visited, color, dimension)
            blank_matrix = np.zeros((dimension, dimension))
            main_island, island_size = fillzoneUtils.get_principal_block_recursive(new_state, blank_matrix, 0, 0, color, 0,
                                                                         dimension)
            if not fillzoneUtils.is_insignificant_move(actual_node.visited, main_island, dimension):
                new_node = node.Node(new_state, main_island, actual_node.cost + movement_cost,
                                     actual_node, color, island_size)
                next_node, border_nodes, total_nodes = dfs_search_fill_zone(new_node, dimension, border_nodes_dfs + 1,
                                                                  total_nodes_dfs + 1)
                if next_node is not None:
                    return next_node, border_nodes, total_nodes
        else:
            border_nodes_dfs = border_nodes_dfs - 1




def a_search_fill_zone(root, dimension, heuristic):
    queue = priorityQueue.PriorityQueue()
    queue.insert(root)

    if fillzoneUtils.is_goal(root, dimension):
        return root, 1, 1

    total_nodes = 1
    border_nodes = 1
    gained_blocks = 0
    while not queue.isEmpty():
        actual_node = queue.pop()
        border_nodes = border_nodes - 1
        if fillzoneUtils.is_goal(actual_node, dimension):
            #print('GOAL ACHIVED')
            return actual_node, border_nodes, total_nodes
        
        # por cada color veo como queda la matriz al escogerlo
        for color in range(colors):
            if color != actual_node.color:
                new_state = fillzoneUtils.change_color(np.copy(actual_node.state), actual_node.visited, color,
                                                       dimension)
                gained_blocks += 1
                blank_matrix = np.zeros((dimension, dimension))
                main_island, island_size = fillzoneUtils.get_principal_block_recursive(new_state, blank_matrix, 0, 0, color, 0,
                                                                             dimension)
                if actual_node.island_size < island_size:
                    new_node = node.Node(new_state, main_island, actual_node.cost + movement_cost,
                                         actual_node, color, island_size)
                    if heuristic == 1:
                        heuristic_val = heuristics.heuristic1(new_node, dimension, colors)
                    elif heuristic == 2:
                      heuristic_val = heuristics.heuristic2(new_state, new_node, dimension)
                    else:
                       heuristic_val = heuristics.heuristic3(new_node, dimension)

                    new_node.set_value(heuristic_val)
                    total_nodes = total_nodes + 1
                    border_nodes = border_nodes + 1

                    if heuristic_val == 0:
                        return new_node, border_nodes, total_nodes
                    queue.insert(new_node)


def greedy_fill_zone(root, dimension, heuristic):
    current = root
    total_nodes = 1
    border_nodes = 0
    gained_blocks = 0

    while not fillzoneUtils.is_goal(current, dimension):
        queue = priorityQueueGreedy.PriorityQueue()
        # por cada color veo como queda la matriz al escogerlo
        for color in range(colors):
            if color != current.color:
                aux = np.copy(current.state)
                new_state = fillzoneUtils.change_color(aux, current.visited, color, dimension)
                gained_blocks += 1                
                blank_matrix = np.zeros((dimension, dimension))
                main_island, island_size = fillzoneUtils.get_principal_block_recursive(new_state, blank_matrix, 0, 0, color, 0,
                                                                             dimension)
                if not fillzoneUtils.is_insignificant_move(current.visited, main_island, dimension):
                    new_node = node.Node(new_state, main_island, current.cost + movement_cost,
                                         current, color, island_size)

                    if heuristic == 1:
                        heuristic_val = heuristics.heuristic1(new_node, dimension, colors)
                    elif heuristic == 2:
                      heuristic_val = heuristics.heuristic2(new_state, new_node, dimension)
                    else:
                       heuristic_val = heuristics.heuristic3(new_node, dimension)

                    new_node.set_value(heuristic_val)
                    queue.insert(new_node)
                    total_nodes = total_nodes + 1
                    border_nodes = border_nodes + 1
        current = queue.pop()
        border_nodes = border_nodes - 1

    return current, border_nodes, total_nodes


def run_all():
    #[[costo,nodos,nodos frontera,tiempo],[costo,nodos,nodos frontera,tiempo],...]
    prints = ["A* HEURISTIC 1", "A* HEURISTIC 2", "A* HEURISTIC 3",
                "GREEDY HEURISTIC 1", "GREEDY HEURISTIC 2", "GREEDY HEURISTIC 3", "DFS", "BFS"]
    print_len = prints.__len__()

    info = [[0 for _ in range(4)] for _ in range(print_len)]

    a_search_1=[]
    a_search_2=[]
    a_search_3=[]
    greedy_1=[]
    greedy_2=[]
    greedy_3=[]
    dfs=[]
    bfs=[]
    for _ in range(TRIES):
        for dimension in dim:
            random_matrix = np.random.randint(0, colors, (dimension, dimension))

            print("--------------------------------------------------------------")
            print("NxN = " + str(dimension) + " x "+ str(dimension))
            print("M = " + str(colors))
            print()
            print()
            print(random_matrix)

            visited = np.zeros((dimension, dimension))

            main_island, island_size = fillzoneUtils.get_principal_block_recursive(random_matrix, visited, 0, 0,
                                                                        random_matrix[0][0], 0, dimension)

            root = node.Node(random_matrix, main_island, 0, None,
                            random_matrix[0][0], island_size)


        
            goals = [None] * print_len
            border_nodes = [0] * print_len
            total_nodes = [0] * print_len
            total_times = [0] * print_len

            
            iteration = 0
            timer = time.time()
            goals[iteration], border_nodes[iteration], total_nodes[iteration] = a_search_fill_zone(root, dimension, 1)
            total_times[iteration] = time.time() - timer
            info[iteration][0] += goals[iteration].cost
            info[iteration][1] += border_nodes[iteration]
            info[iteration][2] += total_nodes[iteration]
            info[iteration][3] += total_times[iteration]
            a_search_1.append([goals[iteration].cost, border_nodes[iteration], total_nodes[iteration], total_times[iteration]])
            iteration += 1

            timer = time.time()
            goals[iteration], border_nodes[iteration], total_nodes[iteration] = a_search_fill_zone(root, dimension, 2)
            total_times[iteration] = time.time() - timer
            info[iteration][0] += goals[iteration].cost
            info[iteration][1] += border_nodes[iteration]
            info[iteration][2] += total_nodes[iteration]
            info[iteration][3] += total_times[iteration]
            a_search_2.append([goals[iteration].cost, border_nodes[iteration], total_nodes[iteration], total_times[iteration]])
            iteration += 1
            
            timer = time.time()
            goals[iteration], border_nodes[iteration], total_nodes[iteration] = a_search_fill_zone(root, dimension, 3)
            total_times[iteration] = time.time() - timer
            info[iteration][0] += goals[iteration].cost
            info[iteration][1] += border_nodes[iteration]
            info[iteration][2] += total_nodes[iteration]
            info[iteration][3] += total_times[iteration]
            a_search_3.append([goals[iteration].cost, border_nodes[iteration], total_nodes[iteration], total_times[iteration]])
            iteration += 1

            timer = time.time()
            goals[iteration], border_nodes[iteration], total_nodes[iteration] = greedy_fill_zone(root, dimension, 1)
            total_times[iteration] = time.time() - timer
            info[iteration][0] += goals[iteration].cost
            info[iteration][1] += border_nodes[iteration]
            info[iteration][2] += total_nodes[iteration]
            info[iteration][3] += total_times[iteration]
            greedy_1.append([goals[iteration].cost, border_nodes[iteration], total_nodes[iteration], total_times[iteration]])
            iteration += 1

            timer = time.time()
            goals[iteration], border_nodes[iteration], total_nodes[iteration] = greedy_fill_zone(root, dimension, 2)
            total_times[iteration] = time.time() - timer
            info[iteration][0] += goals[iteration].cost
            info[iteration][1] += border_nodes[iteration]
            info[iteration][2] += total_nodes[iteration]
            info[iteration][3] += total_times[iteration]
            greedy_2.append([goals[iteration].cost, border_nodes[iteration], total_nodes[iteration], total_times[iteration]])
            iteration += 1

            timer = time.time()
            goals[iteration], border_nodes[iteration], total_nodes[iteration] = greedy_fill_zone(root, dimension, 3)
            total_times[iteration] = time.time() - timer
            info[iteration][0] += goals[iteration].cost
            info[iteration][1] += border_nodes[iteration]
            info[iteration][2] += total_nodes[iteration]
            info[iteration][3] += total_times[iteration]
            greedy_3.append([goals[iteration].cost, border_nodes[iteration], total_nodes[iteration], total_times[iteration]])
            iteration += 1
            
            timer = time.time()
            goals[iteration], border_nodes[iteration], total_nodes[iteration] = dfs_search_fill_zone(root, dimension=dimension)
            total_times[iteration] = time.time() - timer
            info[iteration][0] += goals[iteration].cost
            info[iteration][1] += border_nodes[iteration]
            info[iteration][2] += total_nodes[iteration]
            info[iteration][3] += total_times[iteration]
            dfs.append([goals[iteration].cost, border_nodes[iteration], total_nodes[iteration], total_times[iteration]])
            iteration += 1

            # timer = time.time()
            # # Comentado dado que tarda mucho
            # goals[iteration], border_nodes[iteration], total_nodes[iteration] = bfs_search_fill_zone(root, dimension=dimension)
            # total_times[iteration] = time.time() - timer
            # info[iteration][0] += goals[iteration].cost
            # info[iteration][1] += border_nodes[iteration]
            # info[iteration][2] += total_nodes[iteration]
            # info[iteration][3] += total_times[iteration]
            # bfs.append([goals[iteration].cost, border_nodes[iteration], total_nodes[iteration], total_times[iteration]])

# - 1 para que no me imprima bfs
    for i in range(print_len):
        print(prints[i])
        if (i==0):
            print(a_search_1)
        elif (i==1):
            print(a_search_2)
        elif (i==2):
            print(a_search_3)
        elif (i==3):
            print(greedy_1)
        elif (i==4):
            print(greedy_2)
        elif (i==5):
            print(greedy_3)
        elif (i==6):
            print(dfs)
        else:
            # print(bfs)
            continue

        print('Average Total Cost: ' + str(info[i][0]/TRIES))
        print('Average Expanded Nodes: ' + str(info[i][2]/TRIES))
        print('Average Border Nodes: ' + str(info[i][1]/TRIES))
        print('Average Time: '+ str(info[i][3]/TRIES))
        print()
        print()
    

if __name__ == '__main__':
    run_all()
