# -*- coding: utf-8 -*-
"""
Created on Sat May 27 15:49:56 2023

@author: hqh
"""

import pandas as pd  
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn import metrics
import xgboost as xgb,numpy as np
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
dataset = pd.read_excel(r"D:\Desktop\6013.xlsx")#文件路径
X = dataset.iloc[:,2:11].values
y = dataset.iloc[:,11].values
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#网格搜索最佳参数
param_grid = {'max_depth': [2, 3, 4, 5],
              'n_estimators': [20, 30, 50, 100],
              'learning_rate': [0.1, 0.2, 0.3],
              "gamma":[0.0, 0.1],
              "reg_alpha":[0.01, 0.1],
              "reg_lambda":[ 0.01, 0.1],
              "min_child_weight": [1,2]}


gsearch1 = GridSearchCV(estimator=XGBRegressor(seed=27), param_grid=param_grid, cv=5)
gsearch1.fit(X_train, y_train)
print("best_score_:",gsearch1.best_params_,gsearch1.best_score_)
#用最佳参数进行预测
y_predict1 =gsearch1.predict(X_train)
y_predict2 = gsearch1.predict(x_test)
#评价指标
MAE1=metrics.mean_absolute_error(y_train,y_predict1)
MAE2=metrics.mean_absolute_error(y_test,y_predict2)
print('MAE={}'.format(MAE1))
print('MAE={}'.format(MAE2))
RMSE1=np.sqrt(metrics.mean_squared_error(y_train,y_predict1)) 
RMSE2=np.sqrt(metrics.mean_squared_error(y_test,y_predict2))  # RMSE
print('RMSE={}'.format(RMSE1))
print('RMSE={}'.format(RMSE2))
R1=metrics.r2_score(y_train,y_predict1)
R2=metrics.r2_score(y_test,y_predict2)
print('R1={}'.format(R1))
print('R2={}'.format(R2))
