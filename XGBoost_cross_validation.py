# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 16:09:17 2023

@author: felip
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np  
import xgboost as xgb



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
    
    regressor = xgb.XGBRegressor(max_depth=8,
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