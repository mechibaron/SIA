# SIA
## TP 5
Trabajos Prácticos 

En primer lugar hay que instalar:
- keras
- matplotlib
- numpy
- tensorflow

# Ejercicio 1
Luego para el ejercicio 1 es necesario completar config.json en la carpeta 'ej1' con momentum (true or false, nosotros decidimos utlizar el momentum en true para una mejor optimizacion), eta y epochs deseadas y situarse en ej1_a.py o ej1_b.py segun corresponda y correrlo

Es importante aclarar que en ambos items es posible modificar las capas accediendo al codigo ej1_i.py siendo i el item a y b segun corresponda.
Para hacerlo hay que cambiar los parametros de layers y su arreglo que se encuentran al principio de los archivos.
Por ejemplo se dejan planteados como:
```
layer1 = Layer(20, 35, activation="tanh")
layer2 = Layer(10, activation="tanh")
layer3 = Layer(2, activation="tanh")
layer4 = Layer(10, activation="tanh")
layer5 = Layer(20, activation="tanh")
layer6 = Layer(35, activation="tanh")

layers = [layer1, layer2, layer3, layer4, layer5, layer6] 
```
Lo que significa una arquitectura de tipo -> 35-20-10-2-10-20-35

# Ejercicio 2
En el caso del ejercicio 2 situarse en ej2.py en la carpeta 'ej2' y correrlo

Si se desea modificar las epocas es necesario acceder al archivo 'variational_autoencoder.py' y modificar epochs en la linea 21