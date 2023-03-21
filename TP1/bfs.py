import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from utils import COLORS_COUNT, COLUMN_COUNT, FIRST_COLUMN, FIRST_ROW, ROW_COUNT, belong_to,fill_zone, fill_zone_grid, fill_zone_win, fill_zone_win_grid, get_neighbours_grid, get_principal_block,get_neighbours, colors, get_principal_block_grid

def bfs_algorithm(self):
    inicio = time.time()
    # win = fill_zone_win(self, self.grid[ROW_COUNT-2][COLUMN_COUNT-1])
    count_cost = 0
    expanded_nodes_count=0
    border_nodes=0
    first_color = self.grid[ROW_COUNT - 2][0]
    principal_block = get_principal_block(self)

    color_neighbours=[]

    for block in principal_block:
        color_neighbours.extend(get_neighbours(self, block, principal_block, color_neighbours))
        
    #Cuento los nodos frontera => todos los vecinos de bloque principal 
    border_nodes+=color_neighbours.__len__()

    original_grid = self.grid
    # [costo, nodos expandidos, nodos frontera]
    # TODO: MODIFICAR CUANDO SE CAMBIAN LOS COLORES
    colors_amount= [[0,0,0],[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]]
    for color_i in range(COLORS_COUNT):
        if(colors[color_i] != first_color):
            colors_amount[color_i] = check_color(original_grid, colors[color_i], first_color)

    min_value = colors_amount[0][0]
    index = 0
    for i in range(COLORS_COUNT):
        if(colors_amount[i][0]< min_value):
            min_value = colors_amount[i][0]
            index = i


    fin = time.time()
    return [colors_amount[index][0], colors_amount[index][1], colors_amount[index][2],  fin-inicio]

def check_color(grid, selected_color, first_color):
    color_neighbours=[]
    expanded_nodes = []
    border_nodes = 0
    
    principal_block = get_principal_block_grid(grid)

    for block in principal_block:
        color_neighbours.extend(get_neighbours_grid(grid, block, principal_block, color_neighbours))
    for neighbour in color_neighbours:
        if ((neighbour[0] == selected_color) and (belong_to(expanded_nodes, neighbour[3][0], neighbour[3][1], neighbour[3][2]))== False):
            expanded_nodes.append(neighbour[3])

    fill_zone_grid(grid, selected_color, first_color)
    if(fill_zone_win_grid(grid, grid[FIRST_ROW][FIRST_COLUMN])):
        # [costo, nodos expandidos, nodos frontera]
        return [1, expanded_nodes.__len__(), 0];
    else:
        for color_i in range(COLORS_COUNT):
            if(colors[color_i] != first_color):
                check = check_color(grid, colors[color_i], selected_color)
                return [1+check[0], expanded_nodes.__len__()+check[1], border_nodes + check[2]]
            else:
                continue
    # ERROR