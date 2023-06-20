# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:16:52 2020

@author: marco
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np  



# Regressor
from sklearn.neural_network import MLPRegressor


# Divisão dos dados para validação cruzada
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 3)
scores = []
maes = []
mses = []
rmses = []
mapes = []
for indice_treinamento, indice_teste in kfold.split(previsores,
                                                    np.zeros(shape=(previsores.shape[0], 1))):    
    regressor = MLPRegressor(activation='tanh',
                         max_iter=300,
                         verbose=True,
                         hidden_layer_sizes = (10,10,10),
                         random_state=0)
    
    #  Treinamento
    regressor.fit(previsores[indice_treinamento], objetivo[indice_treinamento,0])
    
    previsoes = regressor.predict(previsores[indice_teste])
    previsoes = scaler_y.inverse_transform(previsoes.reshape(-1,1))
    objetivo_escala_original = scaler_y.inverse_transform(objetivo[indice_teste,0].reshape(-1,1))
    
    score = metrics.r2_score(objetivo_escala_original, previsoes)
    mae = metrics.mean_absolute_error(objetivo_escala_original, previsoes)
    mse = metrics.mean_squared_error(objetivo_escala_original, previsoes)
    rmse = np.sqrt(metrics.mean_squared_error(objetivo_escala_original, previsoes))
    mape = mean_absolute_percentage_error(objetivo_escala_original, previsoes)

    scores.append(score)
    maes.append(mae)
    mses.append(mse)
    rmses.append(rmse)
    mapes.append(mape)


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

mapes = np.asarray(mapes)
mape_final_medio = mapes.mean()
mape_final_desvio_padrao = mapes.std()

################## Gráfico de resíduos #######################################


import numpy as np
import seaborn as sns
sns.set_style("whitegrid")
sns.despine(top=True, right=False, left=False, bottom=False, offset=None, trim=False)

previsoes_treinamento = regressor.predict(previsores[indice_treinamento])
previsoes_treinamento = scaler_y.inverse_transform(previsoes_treinamento.reshape(-1,1))
objetivo_treinamento = scaler_y.inverse_transform(objetivo[indice_treinamento,0].reshape(-1,1))
erros_treinamento = (objetivo_treinamento - previsoes_treinamento) / objetivo_treinamento
erros_teste = (objetivo_escala_original - previsoes) / objetivo_escala_original


# Plot da relação do erro com cada previsão
ax = sns.residplot(x=objetivo_treinamento, y=previsoes_treinamento, lowess=False)
ax = sns.residplot(x=objetivo_escala_original, y=previsoes, lowess=False)
ax.legend(['Teste','Treinamento'],loc="upper right", fontsize=12,fancybox=True, framealpha=1, shadow=True, borderpad=1)
ax.set_xlabel("Imóvel",fontsize=15)
ax.set_ylabel("Desvio Relativo",fontsize=15)
ax.tick_params(labelsize=15)

plt.scatter(x=objetivo_treinamento, y=previsoes_treinamento)
plt.scatter(x=objetivo_escala_original, y=previsoes)
plt.plot(previsoes_treinamento, previsoes_treinamento)


# histograma dos residuos
ax = sns.distplot(erros_treinamento)
ax = sns.distplot(erros_teste)
ax.legend(['Treinamento','Teste'],loc="upper right", fontsize=12,fancybox=True, framealpha=1, shadow=True, borderpad=1)
ax.set_xlabel("Desvio Relativo",fontsize=15)
ax.set_ylabel("Densidade",fontsize=15)
ax.tick_params(labelsize=15)
