# -*- coding: utf-8 -*-
"""
Created on may 18 17:53:31 2023

@author: Felippe
"""

import pandas as pd

################## #Pré-processamento dos dados ################## 

#Leitura dos dados
base = pd.read_csv('output.csv')
base.describe()

# Procurando valores inconsistentes
# Não há valores inconsistentes

# Procurando as colunas que possuem algum valor faltante
# pd.isnull(base).any()

# Separando dados em previsores e classes
cols_previsores = ['carat','cut','color','clarity','depth','table']
#cols_previsores = ['carat','color','clarity','depth','table']
# Não usarei:x,y,z
cols_objetivo = ['price']

previsores = base[cols_previsores]
objetivo = base[cols_objetivo]
   
# Transforma as variáveis categóricas em valores numéricos     
from sklearn.preprocessing import LabelEncoder
labelencoder_previsores = LabelEncoder()
previsores.loc[:, 'color'] = labelencoder_previsores.fit_transform(previsores.loc[:, 'color'])
previsores.loc[:, 'clarity'] = labelencoder_previsores.fit_transform(previsores.loc[:, 'clarity'])

#from sklearn.preprocessing import LabelEncoder
labelencoder_cut = LabelEncoder()
categorias = ["'Fair'","'Good'","'Very Good'","'Premium'","'Ideal'"]
labelencoder_cut.fit(categorias)
previsores.loc[:, 'cut'] = labelencoder_cut.fit_transform(previsores.loc[:, 'cut'])
del categorias


# Padronização dos dados
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)



# padronização para redes neurais (variavel objetivo)
# scaler_y = StandardScaler()
# objetivo = scaler_y.fit_transform(objetivo)


# Separando em base de testes e treinamento (usando 25% para teste)
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, objetivo_treinamento, objetivo_teste = train_test_split(previsores, 
                                                                                                  objetivo, 
                                                                                            test_size=0.25, 
                                                                                                  random_state=0)

################ regressao linear ###################33333
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()

# #treinamento
# regressor.fit(previsores_treinamento, objetivo_treinamento)

# #previsoes
# previsoes = regressor.predict(previsores_teste)


#####################################################################
