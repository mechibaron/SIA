import numpy as np
import pandas as pd
import arcade
import random
import matplotlib.pyplot as plt
import time

from utils import COLUMN_COUNT, ROW_COUNT, belong_to, belong_to_array, get_principal_block

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
