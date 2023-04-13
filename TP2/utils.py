from queue import Queue
import numpy as np
from csv import reader
import genetic
import colors
import matplotlib.pyplot as plt

xpoints = []
ypoints = []
y_red = []
y_green = []
y_blue = []

def get_colors(path) -> np.ndarray:
  file = open(path)
  csvreader = reader(file)

  colores = []
  for row in csvreader:
    r, g, b = (int(x) for x in row)
    colores.append( (r, g, b) )
  return np.array(colores, dtype=np.uint8)

last_fitness = []

def check_finished(iter_amount, iter, pop, mixes, delta, goal, cut_method, seleccion_method, cross_method):

  aps = np.apply_along_axis(genetic.aptitud, 1, mixes, (goal))
  best_aps = np.max(aps)

  order = np.argsort(aps)
  best = np.flip(mixes[order], axis=0)
  best = best[0]
  

  cut = 0
  if(cut_method == 1):
    if(1-best_aps < delta):
      return plot_function(order,best, best_aps, goal, pop, seleccion_method, cross_method, cut_method)
    return False
  elif(cut_method == 2):
    if(iter >= iter_amount):
      return plot_function(order, best, best_aps, goal, pop, seleccion_method, cross_method, cut_method)
    return False
  else:
    if ( iter >= iter_amount or 1 - best_aps < delta):
      if(iter >= iter_amount):
        cut = 2
      else:
        cut = 3
      return plot_function(order,best, best_aps, goal, pop, seleccion_method, cross_method, cut)
    return False


def plot_function(order,best_color,best_aps, goal, pop, selection_method, cross_method, cut_method):
  #PRINT OUTPUTS
  print("Best mix con aptitud = {}".format(best_aps))
  print(best_color)
  print("Props:")
  pop = np.flip(pop[order], axis=0)
  print(pop[0])
  if(cut_method==1):
    print("A cortado debido a 1-Fitness < DELTA")
  else:
    print("A cortado debido a que se recorrio la totalidad de generaciones")
    
  #Show COLORS
  rgb_values_0 = np.array([best_color]) / 255
  rgb_values_1 = np.array([goal]) / 255
  # crear una figura y un eje
  fig, (ax1, ax2) = plt.subplots(1, 2)
  # crear una barra de colores usando la matriz de valores RGB
  ax1.imshow([rgb_values_0])
  ax2.imshow([rgb_values_1])
  # configurar el eje para que muestre las etiquetas de los colores 
  best_color_title = 'Best Color: ' + str(best_color.astype(int))
  ax1.set_title(best_color_title)
  ax1.set_yticks([])
  ax1.set_xticks([])
  goal_color_title = 'Goal Color: ' + str(goal)
  ax2.set_title(goal_color_title)
  ax2.set_yticks([])
  ax2.set_xticks([])

  selection = "Boltzmann"
  if(selection_method == 1):
    selection = "Roulette"
  elif(selection_method == 2):
    selection = "Elite"
  elif(selection_method ==3):
    selection = "Tourney"
  cross = "Uniform"
  if(cross_method == 1):
    cross = "Simple"
  elif(cross_method == 2):
    cross = "Double" 
  title = "Selección " + selection + ", Cruza " + cross
  fig.suptitle(str(title))
  plt.show()

  return True

def get_mixes(rgbp):
  mixes = []
  for i in range(len(rgbp)):
    mix = colors.mix_colors(rgbp[i])
    mixes.append(mix)
  return np.array(mixes)

def get_rgbp(rgbs, p):
  rgbps = []
  s = len(p[0])
  for i in range(len(p)):
    rgbp = np.concatenate((rgbs, p[i].reshape(1, s).T), axis=1)
    rgbps.append(rgbp)
  return np.array(rgbps)
