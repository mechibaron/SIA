import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np



def get_pca_first_component():
    return pca.components_[0]

# def principal_components():
# input_names, inputs, categories = utils.import_data('data/europe.csv')
data = pd.read_csv('data/europe.csv')
variables = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']

#selecciono las columnas especificadas arriba y las aigno a X
X = data[variables]

#estandarizo los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#tengo que hacer el PCA 
pca = PCA()
X_pca = pca.fit_transform(X)

#ahora tengo que sacar las varianzas explicadas
#explained_variance_ratio: porcentaje de varianza explicado por cada uno de los 
#componentes seleccionados.
explained_variances = pca.explained_variance_ratio_

loadings = pca.components_[0]
#Tengo que sacar la carga de variable en la primer componente principal
# loadings = loadings.T
loadings = get_pca_first_component()

# components_df = pd.DataFrame(data=loadings.reshape(1, -1), columns=variables)


# #grafico  boxplot
# plt.bar(range(1,len(explained_variances)+1), explained_variances)
# plt.xlabel('Componentes Principales')
# plt.ylabel('Varianza Explicada')
# plt.show()
# components_df = pd.DataFrame(data=loadings, columns=['PC1', 'PC2'])


# Graficar el biplot de la primer componente principal
plt.figure(figsize=(5, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)

for i in range(len(variables) - 1):
    plt.arrow(0, 0, loadings[i] * 3, loadings[i] * 3, head_width=0.01, head_length=0.01, fc='red', ec='red')
plt.text(loadings[i] * 0.06, loadings[i] * 0.06, variables, color='b', ha='center')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.xlim(-0.3, 0.3)
plt.ylim(-0.3, 0.3)
plt.grid()
plt.show()

