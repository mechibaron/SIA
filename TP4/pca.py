import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns 

def barplot_x(X_standard):
        X_n = X_standard
        # area_x=[fila[0] for fila in X]
        # gdp_x=[fila[1] for fila in X]
        # inf_x=[fila[2] for fila in X]
        # life_x=[fila[3] for fila in X]
        # mil_x=[fila[4] for fila in X]
        # pop_x=[fila[5] for fila in X]
        # unem_x=[fila[6] for fila in X]

        area_xn=[fila[0] for fila in X_n]
        gdp_xn=[fila[1] for fila in X_n]
        inf_xn=[fila[2] for fila in X_n]
        life_xn=[fila[3] for fila in X_n]
        mil_xn=[fila[4] for fila in X_n]
        pop_xn=[fila[5] for fila in X_n]
        unem_xn=[fila[6] for fila in X_n]

        # dfx= {'Area': area_x, 'GDP': gdp_x,'Inflation':inf_x,'Life Expect':life_x, 'Military': mil_x, 'Population Growth': pop_x, 'Unemployment': unem_x}
        # dfx_data=pd.DataFrame(data=dfx, index = None)
        dfxn= {'Area': area_xn, 'GDP': gdp_xn,'Inflation':inf_xn,'Life Expect':life_xn, 'Military': mil_xn, 'Population Growth': pop_xn, 'Unemployment': unem_xn}
        dfxn_data=pd.DataFrame(data=dfxn, index = None)

        # plt.figure(figsize=(25,13))
        # plt.xlabel('Features',fontsize=15) 
        # plt.ylabel('Value',fontsize=15)
        # plt.title(('Non-Standarized Inputs'))
        # dfx_data.boxplot(column=['Area', 'GDP', 'Inflation','Life Expect','Military','Population Growth','Unemployment'])
        # plt.show()

        plt.figure(figsize=(25,13))
        plt.xlabel('Features',fontsize=15) 
        plt.ylabel('Value',fontsize=15)
        plt.title(('Standarized Inputs'))
        dfxn_data.boxplot(column=['Area', 'GDP', 'Inflation','Life Expect','Military','Population Growth','Unemployment'])
        plt.show()

        return None

df = pd.read_csv("data/europe.csv", 
                 names=['Country', 'Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment'], skiprows=[0])

features = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']
# Separating countries
countries = df['Country'].tolist()

# Separating out the features
x = df.loc[:, features].values

# Separating out the target
y = df.loc[:,['Country']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)
barplot_x(x)

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['Country']]], axis = 1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

for i in range(len(countries)):
    indicesToKeep = finalDf['Country'] == countries[i]
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , s = 50)
    # ax.text(finalDf.loc[indicesToKeep, 'principal component 1']
    #            , finalDf.loc[indicesToKeep, 'principal component 2']
    #            , countries[i])
    
# score = principalComponents[:,0:2]
coeff = np.transpose(pca.components_[0:2, :])
n = coeff.shape[0]
for i in range(n):
    plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
    plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, features[i], color = 'g', ha = 'center', va = 'center')
    
ax.legend(countries)
ax.grid()
plt.show()

# explained_variances = pca.explained_variance_ratio_

# plt.bar(range(1,len(explained_variances)+1), explained_variances)
# plt.xlabel('Componentes Principales')
# plt.ylabel('Varianza Explicada')
# plt.show()
# components_df = pd.DataFrame(data=loadings, columns=['PC1', 'PC2'])
