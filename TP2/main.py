import numpy as np

global goal

import utils
import genetic

rng = np.random.default_rng()

# SELECT METHODS
# selection:
#   elite
#   roulette
#   tourney
#   ranking
# crossover:
#   simple
#   double
#   uniform
selector = genetic.SelectOption.ROULETTE
cross_method = genetic.CrossOption.DOUBLE

def main():
  # palette = utils.get_colors("./colores.csv")
  file_name = input("Insert the name of the file: ")
  palette = utils.get_colors("./" + file_name + ".csv")
  print("palette")
  print(palette.shape)
  print(palette)
  goal = np.array([int(x) for x in input("objective color: ").split(',')], dtype=np.uint8)

  # crossover_probability = input("Insert the probability of crossover: ")
  # mutation_probability = input("Insert the probability of mutation: ")

  # goal = np.array([int(x) for x in sys.argv[1].split(',')], dtype=np.uint8)

  # POBLACION INICIAL
  N = 100
  #se genera un array de la forma [[R1,G1,B1],[R2,G2,B2],...] con valores eleatorios uniformes 
  pop = rng.uniform(0., 1., size=(N, len(palette)))

  end = False
  delta = 0.01
  i = 0 

  while (not end):
    print("iteracion: " + str(i))

    rgbp = utils.get_rgbp(palette, pop)
    mixes = utils.get_mixes(rgbp)

    # check criteria
    end = utils.check_finished(i, pop, mixes, delta, goal)

    # SELECTION
    parents = selector(pop, mixes, genetic.aptitud, N, goal)

    # CROSSOVER
    children = genetic.cross_n(parents, cross_method)

    newpop = children if len(children) == N else np.concatenate((children, parents[:(N - len(children))]), axis=0)

    # MUTATION
    newpop = genetic.mutate_n(newpop)

    pop = newpop

    i += 1



if __name__ == '__main__':
  main()