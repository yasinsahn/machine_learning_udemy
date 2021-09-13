#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2021.09.12
author: yasin sahin
written for constructing decision tree regression model

"""

# importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# importing the data
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

# training decision tree regression for whole data
regressor = DecisionTreeRegressor()
regressor.fit(x,y)

# predicting result for a value
y_pred = regressor.predict([[6.5]])

# visualizing results
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x, y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary ($)')