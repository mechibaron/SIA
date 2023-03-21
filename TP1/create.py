import numpy as np
import pandas as pd
import arcade
import random
import matplotlib.pyplot as plt
import time

# Set how many rows and columns we will have
ROW_COUNT = 21
COLUMN_COUNT = 20

COLORS_COUNT = 4

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
    5: arcade.color.YELLOW,
    6: arcade.color.PURPLE,
    7: arcade.color.ORANGE,
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
                        color=colors[random.randrange(0,COLORS_COUNT)]  
                        self.grid[row].append(color)
                        x = (MARGIN + WIDTH) * column + MARGIN + WIDTH // 2
                        y = (MARGIN + HEIGHT) * row + MARGIN + HEIGHT // 2

                        # Draw the box
                        arcade.draw_rectangle_filled(x, y, WIDTH, HEIGHT, color)# Append a cell
            
            # bfs_cost.append(bfs_algorithm(self))
            dfs_cost.append(dfs_algorithm(self))
            # print('El tiempo en ejecucion bfs vs dfs fue de: ', dfs_cost[i][2]) 
            # print('El numero de nodos expandidos de bfs vs dfs es de: ', dfs_cost[i][1]) 
            # print('El numero de nodos frontera en bfs vs dfs es de: ', dfs_cost[i][3])
            # print('El costo de bfs vs dfs es de: ', dfs_cost[i][0])
        print(dfs_cost)
        # Promedio en TRIES_AMOUNT tiradas
        time = 0
        expanded_nodes=0
        border_nodes=0
        cost = 0
        for trie in dfs_cost:
            cost+=trie[0]
            expanded_nodes+=trie[1]
            time+=trie[2]
            border_nodes+=trie[3]

        print(bfs_cost)
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
        principal_block = get_principal_block(self)

        color_neighbours=[]

        for block in principal_block:
            color_neighbours.extend(get_neighbours(self, block, principal_block, color_neighbours))
        
        #Cuento los nodos frontera => todos los vecinos de bloque principal 
        border_nodes+=color_neighbours.__len__()
        
        # Cuento cuantos hay de cada color

        colors_amount= [[0,0], [1,0], [2,0], [3,0], [4,0], [5,0], [6,0], [7,0]]

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
            elif (neighbour[0] == colors[6]):
                colors_amount[6][1]+=1
            elif (neighbour[0] == colors[7]):
                colors_amount[7][1]+=1

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

def belong_to_array(array, item):
    for i in array:
        if(i == item):
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
def dfs_algorithm(self):
    init = time.time()
    finished = False
    expanded_nodes_count = 0
    border_nodes = 0
    cost = 0
    visited = [[0 for _ in range(COLUMN_COUNT)] for _ in range(ROW_COUNT-1)]

    while finished == False:
        principal_block = get_principal_block(self)

        queue = principal_block
        while visit_all(visited)==False and len(queue) > 0:
            first_color = self.grid[FIRST_ROW][FIRST_COLUMN]
            node = queue.pop(0)

            # si ya visitamos ese nodo continuamos con el sig
            if visited[node[1]][node[2]] == 1:
                continue

            visited[node[1]][node[2]] = 1

            # Busco los vecinos del bloque principal (conjunto frontera)
            diff_color_neighbours = []
            diff_color_neighbours.extend(get_neighbours(self, node, principal_block, diff_color_neighbours))

            border_nodes+= diff_color_neighbours.__len__()

            cost +=1
            # Pintame con el primer color del vecino segun prioridad (Abajo,Derecha,Arriba,Izquierda)
            if diff_color_neighbours.__len__() > 0:
                fill_zone(self, diff_color_neighbours[0][0], first_color)
                
            expanded_nodes = [] 
            for neighbour in diff_color_neighbours:
                if ((neighbour[0] == diff_color_neighbours[0][0]) and (belong_to(expanded_nodes, neighbour[3][0], neighbour[3][1], neighbour[3][2]))== False):
                    expanded_nodes.append(neighbour[3])
                    expanded_nodes_count+=1

            # Me fijo quien es ahora el bloque principal una vez que ya pinte
            priority_queue = [[self.grid[FIRST_ROW][FIRST_COLUMN],FIRST_ROW, FIRST_COLUMN]] #Este arreglo se usa solo para popear
            while priority_queue:
                block = priority_queue.pop(0)
                block_neighbours = get_color_neighbours(self, block[1], block[2], block[0],principal_block)
                priority_queue.extend(block_neighbours)
                principal_block.extend(block_neighbours)
                queue.extend(block_neighbours)

        finished = fill_zone_win(self, self.grid[ROW_COUNT-2][0])

    end = time.time()
    return [cost, expanded_nodes_count, end-init, border_nodes]


def visit_all(visited):
    for row in range(len(visited)):
        for col in range(len(visited[0])):
            if visited[row][col] != 1:
                return False
    return True

def get_principal_block(self):
    principal_block = [[self.grid[FIRST_ROW][FIRST_COLUMN],FIRST_ROW, FIRST_COLUMN]] 
    priority_queue = [[self.grid[FIRST_ROW][FIRST_COLUMN],FIRST_ROW, FIRST_COLUMN]] 

    while priority_queue:
        block = priority_queue.pop(0)
        block_neighbours = get_color_neighbours(self, block[1], block[2], block[0],principal_block)
        priority_queue.extend(block_neighbours)
        principal_block.extend(block_neighbours)    
    return principal_block
# Cant de bloques restantes
def heuristic_1(self):
    principal_block = get_principal_block(self)
    
    return ((ROW_COUNT - 1)*COLUMN_COUNT) - principal_block.__len__()

# Cant de colores restantes
def heuristic_2(self):
    principal_block = get_principal_block(self)

    colors = []
    for row in ROW_COUNT - 1:
        for col in COLUMN_COUNT:
            block = [self.grid[row][col],row,col]
            # Si no esta en el bloque principal y no lo anote ya en el arreglo de colores
            if(belong_to(principal_block, block[0], block[1], block[2]) == False and belong_to_array(colors, block[0] == False) ):
                colors.append(block[0])
    return colors


def main():
    MyGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    arcade.run()
        

if __name__ == "__main__":
    main()