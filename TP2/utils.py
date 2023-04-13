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

  xpoints.append(iter)
  ypoints.append(best_aps)
  # y_red.append(best[0])
  # y_green.append(best[1])
  # y_blue.append(best[2])

  print(best_aps)
  
  print("Best mix con aptitud = {}".format(best_aps))
  print(best)
  print("Props:")
  pop = np.flip(pop[order], axis=0)
  print(pop[0])
  cut = 0
  if(cut_method == 1):
    if(1-best_aps < delta):
      return plot_function(xpoints, ypoints, seleccion_method, cross_method, cut_method)
    return False
  elif(cut_method == 2):
    if(iter >= iter_amount):
      return plot_function(xpoints, ypoints, seleccion_method, cross_method, cut_method)
    return False
  else:
    if ( iter >= iter_amount or 1 - best_aps < delta):
      if(iter >= iter_amount):
        cut = 2
      else:
        cut = 3
      return plot_function(xpoints, ypoints, seleccion_method, cross_method, cut)
    return False


def plot_function(xpoints,ypoints, seleccion_method, cross_method, cut_method):
  # plt.plot(xpoints, ypoints)
  # plt.plot(xpoints, y_red, 'r-')
  # plt.plot(xpoints, y_green, 'g-')
  # plt.plot(xpoints, y_blue, 'b-')
  # plt.show()

  # plt.scatter(xpoints, y_red, s=None, c='Red', cmap='Reds')
  # plt.scatter(xpoints, y_green, s=None, c='Green', cmap='Reds')
  # plt.scatter(xpoints, y_blue, s=None, c='Blue', cmap='Reds')
  # plt.xlabel("iteraciones")
  # plt.ylabel("aptitudes")
  # plt.title("Selección Elite, Cruza Simple")
  # plt.show()
  # print(xpoints)
  # print(ypoints)
  if(cut_method==1):
    print("A cortado debido a 1-Fitness < DELTA")
  else:
    print("A cortado debido a que se recorrio la totalidad de generaciones")
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
