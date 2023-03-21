import numpy as np
import pandas as pd
import arcade
import matplotlib.pyplot as plt
# Set how many rows and columns we will have
ROW_COUNT = 4
COLUMN_COUNT = 3

COLORS_COUNT = 6

FIRST_ROW = ROW_COUNT - 2
FIRST_COLUMN = 0

# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 30 
HEIGHT = 30

# This sets the margin between each cell
# and on the edges of the screen.
MARGIN = 5

TRIES_AMOUNT = 1

# Do the math to figure out our screen dimensions
SCREEN_WIDTH = (WIDTH + MARGIN) * COLUMN_COUNT + MARGIN
SCREEN_HEIGHT = (HEIGHT + MARGIN) * ROW_COUNT + MARGIN
SCREEN_TITLE = "FILL ZONE"

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
    print('You won the game!')
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

                
def contains_color(color_i, neighbours):
    for n in neighbours:
        if(n[0]==colors[color_i]):
            return True
    return False
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


def fill_zone_grid(grid, selected_color, first_color):
        """
        Fills the zone around the given cell with the given color.
        """

        if(first_color!=selected_color):
            it_fill_zone_grid(grid, ROW_COUNT-2,0, first_color, selected_color)
        
def it_fill_zone_grid(grid, row,column, first_color, selected_color):
    win = fill_zone_win_grid(grid, grid[ROW_COUNT-2][COLUMN_COUNT-1])
    if(win):
        return
    
    if not in_grid(row,column):
        return
    
    if grid[row][column] == first_color:
        grid[row][column]=selected_color
        x = (MARGIN + WIDTH) * column + MARGIN + WIDTH // 2
        y = (MARGIN + HEIGHT) * row + MARGIN + HEIGHT // 2
        arcade.draw_rectangle_filled(x, y, WIDTH, HEIGHT, selected_color) # Append a cell
        it_fill_zone_grid(grid, row-1, column, first_color, selected_color)
        it_fill_zone_grid(grid, row+1, column, first_color, selected_color)
        it_fill_zone_grid(grid, row, column-1, first_color, selected_color)
        it_fill_zone_grid(grid, row, column+1, first_color, selected_color)

def fill_zone_win_grid(grid, actual_node):
    for row in range(ROW_COUNT - 1):
        for col in range(COLUMN_COUNT):
            if (grid[row][col]!=actual_node):
                return False
    print('You won the game!')
    return True

def get_neighbours_grid(grid, block,principal_block, color_neighbours):
    dx = [-1, 0, 1, 0]  # cambios en x para obtener los vecinos
    dy = [0, 1, 0, -1]  # cambios en y para obtener los vecinos
    vecinos=[]
    for i in range(4):
        nx = block[1] + dx[i]
        ny = block[2] + dy[i]
        if(in_grid(nx, ny)==True):
            if ((belong_to(principal_block,grid[nx][ny],nx,ny) == False) and (belong_to(color_neighbours,grid[nx][ny],nx,ny) == False)):
                vecinos.append([grid[nx][ny],nx,ny, block])
    return vecinos

def get_principal_block_grid(grid):
    principal_block = [[grid[FIRST_ROW][FIRST_COLUMN],FIRST_ROW, FIRST_COLUMN]] 
    priority_queue = [[grid[FIRST_ROW][FIRST_COLUMN],FIRST_ROW, FIRST_COLUMN]] 

    while priority_queue:
        block = priority_queue.pop(0)
        block_neighbours = get_color_neighbours_grid(grid, block[1], block[2], block[0],principal_block)
        priority_queue.extend(block_neighbours)
        principal_block.extend(block_neighbours)    
    return principal_block

def get_color_neighbours_grid(grid, x, y, color,principal_block):
    dx = [-1, 0, 1, 0]  # cambios en x para obtener los vecinos
    dy = [0, 1, 0, -1]  # cambios en y para obtener los vecinos
    vecinos = []
    for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]
        if((in_grid(nx, ny)==True)):
            if ((grid[nx][ny] == color) and (belong_to(principal_block, grid[nx][ny],nx,ny))==False):
                vecinos.append([grid[nx][ny], nx, ny])
    return vecinos