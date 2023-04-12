import numpy as np
import math 

def distance(c1, c2):
  r1, g1, b1 = c1
  r2, g2, b2 = c2

  d_r = r1 - r2
  d_g = g1 - g2
  d_b = b1 - b2
  
  d = math.sqrt( (d_r**2) + (d_g**2) + (d_b**2) )

  return d
# 255, 102, 153
MAX_DISTANCIA = distance((0,0,0), (255,255,255))

def mix_colors(colors):
    alphas = colors[:, -1]
    rgbs = colors[:, :-1]

    total_weight = np.sum(alphas)

    # 255, 125, 0

    rn = np.sum(rgbs[:, 0] * alphas) / total_weight
    gn = np.sum(rgbs[:, 1] * alphas) / total_weight
    bn = np.sum(rgbs[:, 2] * alphas) / total_weight

    return rn, gn, bn