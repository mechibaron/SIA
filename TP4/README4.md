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
    - epochs
    - exercise

4. Configurar el config del ejercicio que se requiera y correr `main.py`


### Ejercicio 1
    El archivo de configuracion del ejercicio se encuentra bajo el nombre config_ej1.json donde se puede modificar el learning rate y el modelo a utilizar (kohonen u oja). 
    En el caso de seleccionar kohonen, configurar el archivo config_kohonene.json donde se indican radio, k y similitud (puede ser euclidea o exponencial)

### Ejercicio 2
    El archivo de configuracion del ejercicio se encuentra bajo el nombre config_hopfield.json donde se puede configurar:
    - train_letters -> vector de letras de entrenamiento
    - noisy_letter -> letra a la cual se le va a aplicar ruido
    - noise_probability -> probabilidad de ruido a aplicar

## Ejecución 🚀

Para ejecutar el programa, simplemente se debe correr en la terminal el archivo main.py:
```
python3 main.py
```