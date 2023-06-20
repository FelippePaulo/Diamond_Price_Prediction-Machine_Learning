# -*- coding: utf-8 -*-
"""
Created on jun 1 16:42:04 2023

@author: felippe
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import xgboost as xgb

################## Preprocessamento ##################

# Arquivo separado

################## Regressão com XGBoost ##################

regressor = xgb.XGBRegressor(max_depth=8,
                             random_state=0)

# Treinamento
regressor.fit(previsores_treinamento, objetivo_treinamento)

# Teste
previsoes = regressor.predict(previsores_teste)

# Resultados na base de treinamento, para verificar overfitting

################## Avaliação dos resultados ##################

score = regressor.score(previsores_teste, objetivo_teste)
mae = metrics.mean_absolute_error(objetivo_teste, previsoes)
mse = metrics.mean_squared_error(objetivo_teste, previsoes)
rmse = np.sqrt(metrics.mean_squared_error(objetivo_teste, previsoes))

print('Score:', score)
print('Mean Absolute Error:', mae)
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)