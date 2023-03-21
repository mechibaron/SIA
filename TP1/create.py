import numpy as np
import pandas as pd
import arcade
import random
import matplotlib.pyplot as plt
from bfs import bfs_algorithm
# from dfs import dfs_algorithm

from utils import COLORS_COUNT, COLUMN_COUNT, FIRST_COLUMN, FIRST_ROW, HEIGHT, MARGIN, ROW_COUNT, SCREEN_HEIGHT, SCREEN_TITLE, SCREEN_WIDTH, TRIES_AMOUNT, WIDTH, fill_zone, colors



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
            # print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA ', i)
            bfs_cost.append(bfs_algorithm(self))
            # dfs_cost.append(dfs_algorithm(self))
            # print('El tiempo en ejecucion bfs vs dfs fue de: ', dfs_cost[i][2]) 
            # print('El numero de nodos expandidos de bfs vs dfs es de: ', dfs_cost[i][1]) 
            # print('El numero de nodos frontera en bfs vs dfs es de: ', dfs_cost[i][3])
            # print('El costo de bfs vs dfs es de: ', dfs_cost[i][0])
        print(bfs_cost)
        # Promedio en TRIES_AMOUNT tiradas
        time = 0
        expanded_nodes=0
        border_nodes=0
        cost = 0
        # costs, expanded, frontier, fin-inicio
        for trie in bfs_cost:
            # cost+=trie[0]
            print(trie[0])
            expanded_nodes+=trie[1]
            border_nodes+=trie[2]
            time+=trie[3]

        print(bfs_cost)
        print('El tiempo promedio es de: ', time / TRIES_AMOUNT) 
        print('El numero de nodos expandidos promedio es de: ', expanded_nodes / TRIES_AMOUNT) 
        print('El numero de nodos frontera promedio es de: ', border_nodes/ TRIES_AMOUNT) 
        # print('El costo promedio es de: ', cost / TRIES_AMOUNT) 

        # plt.figure(figsize=(25,5))
        # plt.plot(bfs_cost,marker ="o")
        # plt.title('Distribución de costos a lo largo de 100 intentos')
        # plt.ylim([10,40])
        # plt.ylabel('Costo')
        # plt.xlabel('Número de intentos')
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

        
       



def main():
    MyGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    arcade.run()
        

if __name__ == "__main__":
    main()