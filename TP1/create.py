import numpy as np
import arcade
import random
from collections import deque
from time import sleep


"""
Array Backed Grid Shown By Sprites

Show how to use a two-dimensional list/array to back the display of a
grid on-screen.

This version makes a grid of sprites instead of numbers. Instead of
interating all the cells when the grid changes we simply just
swap the color of the selected sprite. This means this version
can handle very large grids and still have the same performance.

If Python and Arcade are installed, this example can be run from the command line with:
python -m arcade.examples.array_backed_grid_sprites_2
"""


# Set how many rows and columns we will have
ROW_COUNT = 16
COLUMN_COUNT = 15
COLORS_COUNT = 6

# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 30 
HEIGHT = 30

# This sets the margin between each cell
# and on the edges of the screen.
MARGIN = 5

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
    5:arcade.color.YELLOW
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
        
        bfs_algorithm(self)

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
            print("First color: ", first_color)
            print("Selected color: ", selected_color)
            fill_zone(self, selected_color,first_color)

        
       
def fill_zone(self, selected_color, first_color):
        """
        Fills the zone around the given cell with the given color.
        """

        print('First color:',first_color)
        print('Selected color:',selected_color)
        it_fill_zone(self, ROW_COUNT-2,0, first_color, selected_color)
        
def it_fill_zone(self, row,column, first_color, selected_color):
    win = fill_zone_win(self, self.grid[ROW_COUNT-2][COLUMN_COUNT-1])
    if(win):
        print('Gane')
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
    win = fill_zone_win(self, self.grid[ROW_COUNT-2][COLUMN_COUNT-1])
    count_cost = 0
    while not win:
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
            neighbours = get_neighbours(self, block[1], block[2],color_neighbours)
            for n in neighbours:
                color_neighbours.append(n)

        # Elimino repetidos de color_neighbours
        # neighbours = []
        # for neighbour in color_neighbours:
        #     if neighbour not in neighbours:
        #         neighbours.append(neighbour)
        
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
        count_cost +=1

        fill_zone(self, colors[color_selected], first_color)
        print('Volvi y pruebo el win')
        win = fill_zone_win(self, self.grid[ROW_COUNT-2][COLUMN_COUNT-1])

    print("El costo fue de: ",count_cost)

def get_color_neighbours(self, x, y, color,principal_block):
    dx = [-1, 0, 1, 0]  # cambios en x para obtener los vecinos
    dy = [0, 1, 0, -1]  # cambios en y para obtener los vecinos
    vecinos = []
    for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]
        if in_grid(nx, ny) and self.grid[nx][ny] == color and belong_to(principal_block, self.grid[nx][ny])==False:
            print('Encontre un vecino con mi color')
            vecinos.append([self.grid[nx][ny], nx, ny])
    return vecinos

def belong_to(principal_block, color_block):
    for block in principal_block:
        if(block[0] == color_block):
            return True
    return False


def fill_zone_win(self, actual_node):
    for row in range(ROW_COUNT - 1):
        for col in range(COLUMN_COUNT):
            if (self.grid[row][col]!=actual_node):
                return False
    return True

# Función para verificar si una posición (x, y) está dentro de la matriz
def in_grid(x, y,):
    return x >= 0 and x < ROW_COUNT - 1 and y >= 0 and y < COLUMN_COUNT

# Función para obtener las posiciones adyacentes en la matriz
def get_neighbours(self, x, y,color_neighbours):
    dx = [-1, 0, 1, 0]  # cambios en x para obtener los vecinos
    dy = [0, 1, 0, -1]  # cambios en y para obtener los vecinos
    vecinos = []
    for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]
        if in_grid(nx, ny) and belong_to(color_neighbours,self.grid[nx][ny]) == False:
            vecinos.append([self.grid[nx][ny], nx, ny])
    return vecinos
    

def main():
    MyGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    arcade.run()

if __name__ == "__main__":
    main()