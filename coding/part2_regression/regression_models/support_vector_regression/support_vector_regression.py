#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2021.09.12
author: yasin sahin
written for constructing support vector regression model

"""

# importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# importing the data
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values
y = y.reshape(len(y),1) # reshaping y to be 2-D variable to use on feature scaling

# scaling features
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x = scaler_x.fit_transform(x) # scaling x values on normal distribution
y = scaler_y.fit_transform(y) # scaling y values on normal distribution

# training SVR model on whole dataset
regressor = SVR(kernel = 'rbf') # support vector regression with rbf kernel
regressor.fit(x,y) # training the model

# predicting single SVR result
y_pred = \
    scaler_y.inverse_transform(regressor.predict(scaler_x.transform([[6.5]])))


# visualizing results
x_grid = np.arange(min(x),max(x),0.1) # gridding for smoother curve
x_grid = x_grid.reshape(len(x_grid),1) # making grid 2-D
plt.scatter(scaler_x.inverse_transform(x),\
            scaler_y.inverse_transform(y), color = 'red')
plt.plot(scaler_x.inverse_transform(x_grid),\
         scaler_y.inverse_transform(regressor.predict(x_grid)), color='blue')
plt.title('Truth of Bluff (Support Vector Regression)')
plt.xlabel('Experience Level')
plt.ylabel('Salary ($)')
plt.show()
