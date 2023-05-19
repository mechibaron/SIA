import numpy as np
import pandas as pd
import json
import utils 
import matplotlib.pyplot as plt

with open('./json/config_oja.json', 'r') as f:
    data = json.load(f)
    learning_rate, epochs = utils.getDataFromFile(data)

    f.close()

df = pd.read_csv("data/europe.csv", 
                 names=['Country', 'Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment'], skiprows=[0])


features = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']
countries = df['Country'].values

countries = df['Country'].tolist()
X = df[features].values

#inicio los pesos aleatoriamente
n_features = X.shape[1]
weights = np.random.randn(n_features)
#normalizo los datos 
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
components = np.dot(X, weights)
# Calculo la matriz de covarianza
cov_matrix = np.cov(X, rowvar=False)

# Calculo los autovalores y autovectores
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Ordeno los autovectores en función de los autovalores
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Calculo las componentes principales
components = np.dot(X, sorted_eigenvectors)


#Regla de oja
for _ in range(epochs):
    for i in range(X.shape[0]):
        x = X[i]
        y = np.dot(x, weights)
        weights += learning_rate * y * (x - y * weights)
        #Normalizar los pesos despues de actualizarlos
        weights /= np.linalg.norm(weights)
        
first_component = weights
sorted_indices = np.argsort(first_component)
sorted_features = [features[i] for i in sorted_indices]

#queria ver los valores no es necesario
# print("Interpretación de la primera componente:")
# for i in range(len(sorted_features)):
#     print(f"{sorted_features[i]}: {first_component[i]: .4f}")


#grafico
fig, ax = plt.subplots(figsize=(10, 6))
ind = np.arange(len(countries))

# Plotear las componentes principales para cada país
positive_values = np.maximum(components[:,0], 0)
negative_values = np.minimum(components [:,0], 0)
ax.bar(ind, positive_values, color='b')
ax.bar(ind, negative_values, color='r')
ax.set_xlabel('País')
ax.set_ylabel('Valor de la componente')
ax.set_title('Componente principal para cada país')
ax.set_xticks(ind)
ax.set_xticklabels(countries, rotation=45)
plt.tight_layout()
plt.show()