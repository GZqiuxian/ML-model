# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 17:16:43 2022

@author: hqh
"""


from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

dataset = pd.read_excel(r"D:\Desktop\6013.xlsx")
X = dataset.iloc[:,0:9].values
y = dataset.iloc[:,9].values
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#网格搜索最佳参数
param_grid = {'max_depth': [2, 4, 7, 9],
              'n_estimators': [100, 200, 300, 400, 500],
              'max_features': [2, 3, 5, 7],
              "min_samples_leaf":[1, 3, 5, 7],
              "min_samples_split":[1, 2, 3]}

random_forest_model = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, cv=5)
random_forest_model.fit(X_train, y_train)
print("best_score_:",random_forest_model.best_params_,random_forest_model.best_score_)
#用最佳参数进行预测
y_predict1 = random_forest_model.predict(X_train)
y_predict2 = random_forest_model.predict(x_test)

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