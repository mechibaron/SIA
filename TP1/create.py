import numpy as np
import pandas as pd
import arcade
import random
import matplotlib.pyplot as plt
import time

# Set how many rows and columns we will have
ROW_COUNT = 16
COLUMN_COUNT = 15
COLORS_COUNT = 6

FIRST_ROW = ROW_COUNT - 2
FIRST_COLUMN = 0

# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 30 
HEIGHT = 30

# This sets the margin between each cell
# and on the edges of the screen.
MARGIN = 5

TRIES_AMOUNT = 100

# Do the math to figure out our screen dimensions
SCREEN_WIDTH = (WIDTH + MARGIN) * COLUMN_COUNT + MARGIN
SCREEN_HEIGHT = (HEIGHT + MARGIN) * ROW_COUNT + MARGIN
SCREEN_TITLE = "Array Backed Grid Buffered Example"

colors = {
    0: arcade.color.WHITE,
    1: arcade.color.RED,
    2: arcade.color.BLUE,
    3: arcade.color.GREEN,
    4: arcade.color.PINK,
    5: arcade.color.YELLOW
}

class MyGame(arcade.Window):
    """
    Main application class.
    """

    def __init__(self, width, height, title):
        """
        Set up the application.
        """
        super().__init__(width, height, title)

        # Set the background color of the window
        self.background_color = arcade.color.BLACK
        bfs_cost= []
        dfs_cost = []
        for i in range(TRIES_AMOUNT):
            self.grid = []
            for row in range(ROW_COUNT):
                self.grid.append([])
                for column in range(COLUMN_COUNT):
                    if row == ROW_COUNT - 1 :
                        if column < COLORS_COUNT:
                            color=colors[column]  
                            self.grid[row].append(color)
                            x = (MARGIN + WIDTH) * column + MARGIN + WIDTH // 2
                            y = (MARGIN + HEIGHT) * row + MARGIN + HEIGHT // 2
                            # Draw the box
                            arcade.draw_rectangle_filled(x, y, WIDTH, HEIGHT, color)  # Append a cell
                    else:
                        color=colors[random.randrange(0,5)]  
                        self.grid[row].append(color)
                        x = (MARGIN + WIDTH) * column + MARGIN + WIDTH // 2
                        y = (MARGIN + HEIGHT) * row + MARGIN + HEIGHT // 2

                        # Draw the box
                        arcade.draw_rectangle_filled(x, y, WIDTH, HEIGHT, color)# Append a cell
            
            # bfs_cost.append(bfs_algorithm(self))
            dfs_cost.append(dfs_algorith(self))
            # print('El tiempo en ejecucion bfs fue de: ', bfs_cost[i][2]) 
            # print('El numero de nodos expandidos de bfs es de: ', bfs_cost[i][1]) 
            # print('El numero de nodos frontera en bfd es de: ', bfs_cost[i][3])
            # print('El costo de bfs es de: ', bfs_cost[i][0])
        
        # Promedio en TRIES_AMOUNT tiradas
        time = 0
        expanded_nodes=0
        border_nodes=0
        cost = 0
        for trie in bfs_cost:
            time+=trie[2]
            expanded_nodes+=trie[1]
            border_nodes+=trie[3]
            cost+=trie[0]

        print('El tiempo promedio es de: ', time / TRIES_AMOUNT) 
        print('El numero de nodos expandidos promedio es de: ', expanded_nodes / TRIES_AMOUNT) 
        print('El numero de nodos frontera promedio es de: ', border_nodes/ TRIES_AMOUNT) 
        print('El costo promedio es de: ', cost / TRIES_AMOUNT) 

        plt.figure(figsize=(25,5))
        plt.plot(bfs_cost,marker ="o")
        plt.title('Distribución de costos a lo largo de 100 intentos')
        plt.ylim([10,40])
        plt.ylabel('Costo')
        plt.xlabel('Número de intentos')
        plt.show()
        # a = [1, 3, 5, 7]
        # b = [11, 2, 4, 19]
  
        # # Plot scatter here
        # plt.bar(a, b)
        
        # c = [1, 3, 2, 1]
        
        # plt.errorbar(a, b, yerr=c, fmt="o", color="r")
        
        # plt.show()
        

    def on_mouse_press(self, x, y, button, modifiers):
        """
        Called when the user presses a mouse button.
        """

        # Convert the clicked mouse position into grid coordinates
        column = int(x // (WIDTH + MARGIN))
        row = int(y // (HEIGHT + MARGIN))

        if row == ROW_COUNT-1 and column < COLORS_COUNT:
            selected_color = colors[column]
            first_color = self.grid[ROW_COUNT - 2][0]
            fill_zone(self, selected_color,first_color)

        
       
def fill_zone(self, selected_color, first_color):
        """
        Fills the zone around the given cell with the given color.
        """

        if(first_color!=selected_color):
            it_fill_zone(self, ROW_COUNT-2,0, first_color, selected_color)
        
def it_fill_zone(self, row,column, first_color, selected_color):
    win = fill_zone_win(self, self.grid[ROW_COUNT-2][COLUMN_COUNT-1])
    if(win):
        return
    
    if not in_grid(row,column):
        return
    
    if self.grid[row][column] == first_color:
        self.grid[row][column]=selected_color
        x = (MARGIN + WIDTH) * column + MARGIN + WIDTH // 2
        y = (MARGIN + HEIGHT) * row + MARGIN + HEIGHT // 2
        arcade.draw_rectangle_filled(x, y, WIDTH, HEIGHT, selected_color) # Append a cell
        it_fill_zone(self, row-1, column, first_color, selected_color)
        it_fill_zone(self, row+1, column, first_color, selected_color)
        it_fill_zone(self, row, column-1, first_color, selected_color)
        it_fill_zone(self, row, column+1, first_color, selected_color)

def bfs_algorithm(self):
    inicio = time.time()
    win = fill_zone_win(self, self.grid[ROW_COUNT-2][COLUMN_COUNT-1])
    count_cost = 0
    expanded_nodes_count=0
    border_nodes=0
    while win==False:
        first_color = self.grid[ROW_COUNT - 2][0]
        principal_block = [[self.grid[ROW_COUNT-2][0], ROW_COUNT - 2, 0]]
        queue=[[self.grid[ROW_COUNT-2][0], ROW_COUNT - 2, 0]]
        while queue:
            block = queue.pop(0)
            block_neighbours = get_color_neighbours(self, block[1], block[2], block[0],principal_block)
            for neighbour in block_neighbours:
                queue.append(neighbour)
                principal_block.append(neighbour)

        color_neighbours=[]

        for block in principal_block:
            neighbours = get_neighbours(self, block, principal_block, color_neighbours)
            for n in neighbours:
                color_neighbours.append(n)
        
        #Cuento los nodos frontera => todos los vecinos de bloque principal 
        border_nodes+=color_neighbours.__len__()
        
        # Cuento cuantos hay de cada color
        colors_amount= [[0,0], [1,0], [2,0], [3,0], [4,0], [5,0]]

        for neighbour in color_neighbours:
            if (neighbour[0] == colors[0]):
                colors_amount[0][1]+=1
            elif (neighbour[0] == colors[1]):
                colors_amount[1][1]+=1        
            elif (neighbour[0] == colors[2]):
                colors_amount[2][1]+=1
            elif (neighbour[0] == colors[3]):
                colors_amount[3][1]+=1
            elif (neighbour[0] == colors[4]):
                colors_amount[4][1]+=1
            elif (neighbour[0] == colors[5]):
                colors_amount[5][1]+=1

        # Veo cual es el color que mas aparece
        color_selected = 0
        color_count = 0
        for color in colors_amount:
            if(color[1]>color_count):
                color_selected = color[0]
                color_count = color[1]

        # Veo que nodos expandi => me fijo de quien es vecino el nodo a pintar y los cuento (tengo en cuenta fijarme si no lo conte antes)
        expanded_nodes = []
        for neighbour in color_neighbours:
            if ((neighbour[0] == colors[color_selected]) and (belong_to(expanded_nodes, neighbour[3][0], neighbour[3][1], neighbour[3][2]))== False):
                expanded_nodes.append(neighbour[3])
                expanded_nodes_count+=1


        count_cost +=1
        fill_zone(self, colors[color_selected], first_color)
        win = fill_zone_win(self, self.grid[ROW_COUNT-2][0])
    fin = time.time()
    return [count_cost, expanded_nodes_count, fin-inicio, border_nodes]

def get_color_neighbours(self, x, y, color,principal_block):
    dx = [-1, 0, 1, 0]  # cambios en x para obtener los vecinos
    dy = [0, 1, 0, -1]  # cambios en y para obtener los vecinos
    vecinos = []
    for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]
        if((in_grid(nx, ny)==True)):
            if ((self.grid[nx][ny] == color) and (belong_to(principal_block, self.grid[nx][ny],nx,ny))==False):
                vecinos.append([self.grid[nx][ny], nx, ny])
    return vecinos

def belong_to(principal_block, color_block,nx,ny):
    for block in principal_block:
        if(block[0] == color_block and block[1]==nx and block[2]==ny):
            return True
    return False

def belong_to_dfs(self, principal_block, color_block,nx,ny):
    for block in principal_block:
        if(self.grid[block[0]][block[1]] == color_block and block[0]==nx and block[1]==ny):
            return True
    return False


def fill_zone_win(self, actual_node):
    for row in range(ROW_COUNT - 1):
        for col in range(COLUMN_COUNT):
            if (self.grid[row][col]!=actual_node):
                return False
    return True

# Función para verificar si una posición (x, y) está dentro de la matriz
def in_grid(x, y):
    if (x >= 0 and x < ROW_COUNT - 1 and y >= 0 and y < COLUMN_COUNT):
        return True
    return False

# Función para obtener las posiciones adyacentes en la matriz
#devuelve en el primero el color, en el segundo valor en ex, el tercero el valor en y el cuarto el padre.
#sin repetidos y desordenados
def get_neighbours(self, block,principal_block, color_neighbours):
    dx = [-1, 0, 1, 0]  # cambios en x para obtener los vecinos
    dy = [0, 1, 0, -1]  # cambios en y para obtener los vecinos
    vecinos=[]
    for i in range(4):
        nx = block[1] + dx[i]
        ny = block[2] + dy[i]
        if(in_grid(nx, ny)==True):
            if ((belong_to(principal_block,self.grid[nx][ny],nx,ny) == False) and (belong_to(color_neighbours,self.grid[nx][ny],nx,ny) == False)):
                vecinos.append([self.grid[nx][ny],nx,ny, block])
    return vecinos
    
# Definimos que la metrica de profundidad refiere a dar un paso para abajo y luego para la derecha 
def dfs_algorith(self):
    init = time.time()
    finished = False
    expanded_nodes_count = 0
    cost = 0
    while finished == False:
        first_color = self.grid[FIRST_ROW][FIRST_COLUMN]
        visited = [[0 for _ in range(COLUMN_COUNT)] for _ in range(ROW_COUNT-1)]
        visited[FIRST_ROW][FIRST_COLUMN] = 1 # declaro el primer bloque como visitado
        principal_block = [[ROW_COUNT - 2, 0]]
        print("principal block: " ,principal_block)
        priority_queue = []
        priority_queue.extend(get_neighbours_sorted(self,[ROW_COUNT - 2, 0],principal_block))
        # print("pq:", priority_queue)
        while visit_all(visited) or len(priority_queue) > 0:
            node = priority_queue.pop(0)

            # si ya visitamos ese nodo continuamos con el sig
            print("node: ",node)
            print(pd.DataFrame(visited))
            if visited[node[0]][node[1]] == 1:
                continue

            visited[node[0]][node[1]] = 1

            # explorar los vecinos de node del mismo color
            block_neighbours = get_color_neighbours_sorted(self,node[0], node[1], self.grid[node[0]][node[1]], principal_block)
            print("block neighbours:",block_neighbours)

            # agrego los bloques del mismo color al bloque principal. 
            principal_block.append(node)
            for neighbour in block_neighbours:
                principal_block.append(neighbour)

            # print("new principal: ", principal_block)

            aux_priority = []
            for block in principal_block:
                neighbours = get_neighbours_sorted(self, block, principal_block) #buscamos todos los adyacentes
                # print(neighbours)
                if len(neighbours) != 0:
                    aux_priority.insert(0,neighbours[0])
            aux_priority += priority_queue
            # print("\t aux:",aux_priority)
            priority_queue = get_no_repeated(visited,aux_priority)
            # priority_queue = get_not_visited(aux_priority)
                
            # marco como visit el nodo en cuestion y los nodos vecinos del mismo color 
            
            cost +=1
            fill_zone(self, self.grid[node[0]][node[1]], first_color)

        #     # gane?
        finished = fill_zone_win(self, self.grid[ROW_COUNT-2][0])
        # print("corte")

        # # buscar los vecinos del nodo en cuestion nodo0 (primer caso: 14 0) -> ponerlos en la pq 
        # # vamos al nodo de la izquierda (nodo1)y explotamos. 
        # # buscar los nodos adyacentes del nodo1 que tengan el mismo color y marcar como un bloque de nodos. (nodos que vamos comiendo)
        # # update la pq con los nuevos nodos (los vecinos de nodo1 del mismo color). La prioridad va a ser del de mas a la izquiera. 
        # # volver a empezar siendo el nodo en cuestion el nodo de mayor prioridad. 

        # # prioridad: Rigth Down Left Up
    end = time.time()
    return [end - init, cost]

# def get_not_visited(priority_queue):
#     to_return = []
#     for block in priority_queue:
#         if type(block) == list and len(block) > 0:
#             for sub_block in block:
#                 if sub_block not in to_return:
#                     to_return.append(sub_block)
def visit_all(visited):
    for row in range(len(visited)):
        for col in range(len(visited[0])):
            if visited[row][col] != 1:
                return False
    return True


def get_neighbours_sorted(self, block, principal_block):
    dx = [1, 0, -1, 0]  # cambios en x para obtener los vecinos
    dy = [0, 1, 0, -1]  # cambios en y para obtener los vecinos
    vecinos=[]
    for i in range(4):
        nx = block[0] + dx[i]
        ny = block[1] + dy[i]
        if(in_grid(nx, ny)==True):
            if ((belong_to_dfs(self,principal_block,self.grid[nx][ny],nx,ny) == False)):
                vecinos.append([nx,ny])
    return vecinos

def get_no_repeated(visited,neighbours):
    to_return = []
    for nei in neighbours:
        if nei not in to_return and visited[nei[0]][nei[1]] != 1:
            to_return.append(nei)
    print(to_return)
    return to_return

def get_color_neighbours_sorted(self, x, y, color, principal_block):
    dx = [1, 0, -1, 0]  # cambios en x para obtener los vecinos
    dy = [0, 1, 0, -1]  # cambios en y para obtener los vecinos
    vecinos = []
    for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]
        if((in_grid(nx, ny)==True)):
            if ((self.grid[nx][ny] == color) and (belong_to_dfs(self,principal_block,self.grid[nx][ny],nx,ny) == False)):
                vecinos.append([nx, ny])
    return vecinos

def main():
    MyGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    arcade.run()
        

if __name__ == "__main__":
    main()

14,0    
# 1 2 2 4 5
# 3 4 5 2 1
# 4 3 2 4 1
# 5 4 3 2 1
# 3 5 4 1 2

