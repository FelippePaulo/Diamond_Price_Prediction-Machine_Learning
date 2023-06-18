# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:54:57 2020

@author: marco
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np  

################## Preprocessamento ################## 

#Leitura dos dados
base = pd.read_csv('house_prices.csv')
base.describe()

# Procurando valores inconsistentes
# Não há valores inconsistentes

# Procurando as colunas que possuem algum valor faltante
pd.isnull(base).any()

# Separando dados em previsores e classes
cols_previsores = ['bedrooms','bathrooms','sqft_living', 'sqft_lot', 
                   'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
                   'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long']
# Não usarei: date, sqft_living15 e sqft_lot15

cols_objetivo = ['price']
previsores = base[cols_previsores]
objetivo = base[cols_objetivo]

# Transforma as variáveis categóricas em valores numéricos   
#  Todas as variáveis são numéricas

# Padronização dos dados
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

#####################################################################
####################### Validação cruzada ###########################
#####################################################################

# Regressor
from sklearn.ensemble import RandomForestRegressor


# Divisão dos dados para validação cruzada
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 3)
scores = []
maes = []
mses = []
rmses = []
for indice_treinamento, indice_teste in kfold.split(previsores,
                                                    np.zeros(shape=(previsores.shape[0], 1))):    
    
    regressor = RandomForestRegressor(n_estimators=100,     # número de árvores
                                  max_features=10, # qtd de caracteristicas
                                  max_depth=15,
                                  random_state=0)
    
    #  Treinamento
    regressor.fit(previsores[indice_treinamento], objetivo.iloc[indice_treinamento,0])
    
    previsoes = regressor.predict(previsores[indice_teste])
    
    score = metrics.r2_score(objetivo.iloc[indice_teste,0], previsoes)
    mae = metrics.mean_absolute_error(objetivo.iloc[indice_teste,0], previsoes)
    mse = metrics.mean_squared_error(objetivo.iloc[indice_teste,0], previsoes)
    rmse = np.sqrt(metrics.mean_squared_error(objetivo.iloc[indice_teste,0], previsoes))

    scores.append(score)
    maes.append(mae)
    mses.append(mse)
    rmses.append(rmse)


######################## Resultado final ########################
# Métricas médias
scores = np.asarray(scores)
score_final_medio = scores.mean()
score_final_desvio_padrao = scores.std()

maes = np.asarray(maes)
mae_final_medio = maes.mean()
mae_final_desvio_padrao = maes.std()

mses = np.asarray(mses)
mse_final_medio = mses.mean()
mse_final_desvio_padrao = mses.std()

rmses = np.asarray(rmses)
rmse_final_medio = rmses.mean()
rmse_final_desvio_padrao = rmses.std()