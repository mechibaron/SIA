import numpy as np

class Fillzone:
  ''' Clase que define un juego de fillzone, tiene su estado y sus acciones para modificar este.'''
  def __init__(self, n, k):
    self.state = np.random.randint(0, k, size=(n, n), dtype='u4')

  def update(self, key):
    '''Cambia la matriz acorde a la tecla ingresada'''
    pass

  def check(self):
    '''Checkea si el estado actual es solucion'''
    pass