# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:17:01 2020

@author: marco
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np  

################## Preprocessamento ################## 

scaler_y = StandardScaler()
objetivo_treinamento = scaler_y.fit_transform(objetivo_treinamento)

################## Regressão com Árvores de Decisão ################## 

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf', 
                 C = 1, 
                 gamma = 'auto', 
                 epsilon = 0.1)

#  Treinamento
regressor.fit(previsores_treinamento, objetivo_treinamento)

# Teste
previsoes = regressor.predict(previsores_teste)

previsoes = scaler_y.inverse_transform(previsoes.reshape(-1,1))

################## Avaliação dos resultados ################## 

score = metrics.r2_score(objetivo_teste, previsoes)
mae = metrics.mean_absolute_error(objetivo_teste, previsoes)
mse = metrics.mean_squared_error(objetivo_teste, previsoes)
rmse = np.sqrt(metrics.mean_squared_error(objetivo_teste, previsoes))

print('Score:', score)  
print('Mean Absolute Error:', mae)  
print('Mean Squared Error:', mse)  
print('Root Mean Squared Error:', rmse)

