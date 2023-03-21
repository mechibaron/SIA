import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from create import fill_zone

from utils import COLUMN_COUNT, ROW_COUNT, belong_to,fill_zone, fill_zone_win, get_principal_block,get_neighbours, colors

def dfs_algorithm(self):
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
