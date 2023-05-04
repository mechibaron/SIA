# SIA
Trabajos Prácticos 

### Sistemas de Inteligencia Artificial - Grupo 1

## Instalación 🛠️

1. Descargar el repositorio en su PC.
2. Instalar las librerias necesarias para ejecutar el programa. Debe ejecutar los siguientes comandos en su terminal
```
pip install numpy
pip install math
```
3. Cargar los datos necesarios en el archivo `config.json`  donde se indican :
    - learning_rate
    - epochs
    - bias
    - type_perceptron => puede ser "escalon", "lineal", "no_lineal" o "multicapa" segun el ejercicio a analizar
4. Configurar el config del ejercicio que se requiera y correr `main.py`


### Ejercicio 1
    El archivo de configuracion del ejercicio se encuentra bajo el nombre ej1_config.json donde se puede modificar la operacion deseada a realizar en el parametro "operation" el cual puede ser AND o XOR

### Ejercicio 2
    El archivo de configuracion del ejercicio se encuentra bajo el nombre ej2_config.json donde se puede modificar la operacion theta deseada a realizar en el parametro "theta" el cual puede ser tanh o sigmoid , y se indica beta

### Ejercicio 3
    El archivo de configuracion que permite cambiar que inciso se resuelve se encuentra bajo el nombre ej3_config.json y se varia el parametro "exercise" en 0, 1 o 2 segun item a, b y c respectivamente

## Configuracion de incisos
    - hiddenLayers = arreglo que indica la cantidad de neuronas en cada hidden layer
    - batch_size = toma valores enteros > 0 ; si vale 1 => el perceptron funciona como perceptron incremental
    - momentum = 0 o 1 indicando si esta activo o no
    - adaptative_eta = 0 o 1 indicando si esta activo o no y se asignan sus valores en:
        * adaptative_k
        * adaptative_inc
        * adaptative_dec

    # item a => el archivo de configuración se encuentra bajo el nombre `ex3config/ej3_1_config.json`
    # item b => el archivo de configuración se encuentra bajo el nombre `ex3config/ej3_2_config.json`
    # item c => el archivo de configuración se encuentra bajo el nombre `ex3config/ej3_3_config.json`