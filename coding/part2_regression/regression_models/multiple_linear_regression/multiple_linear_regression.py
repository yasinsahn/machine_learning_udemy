#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2021.09.11
author: yasin sahin
written for constructing multiple linear regression model

"""

# importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# importing the data
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# encoding categorical state names
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])], \
                       remainder = 'passthrough')
x = np.array(ct.fit_transform(x))

# prepparing test & training sets
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size = 0.2,random_state=0)
    
# training the regression model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test) # evaluating test set
np.set_printoptions(precision=2) # taking maximum of 2 decimals

# printing real and prediction results for test set
print(np.concatenate((y_pred.reshape(len(y_pred),1),\
                      y_test.reshape(len(y_test),1)),1))
