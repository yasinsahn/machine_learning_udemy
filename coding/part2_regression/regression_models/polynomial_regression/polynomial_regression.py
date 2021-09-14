#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2021.09.12
author: yasin sahin
written for constructing polynomial regression model

"""

# importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# importing the data
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

# training linear regression model on whole data
linear_regressor = LinearRegression()
linear_regressor.fit(x,y)

# training polynomial regression model on whole data
polynomial_transformer = PolynomialFeatures(degree=4) # creating polynomial transformer object for specified degree
x_poly = polynomial_transformer.fit_transform(x) # transform linear independent variable to polynomial
linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(x_poly,y) # fit linear regression for polynomial independent variable

# plotting linear regression results
plt.scatter(x, y, color='red') # scatter plot for dataset
plt.plot(x, linear_regressor.predict(x), color='blue') # prediction plot
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary ($)')
plt.show()

# plotting polynomial regression results
x_grid = np.arange(min(x),max(x),0.1) # gridding array to have a smoother curve
x_grid = x_grid.reshape(len(x_grid),1) # reshaping to use in prediction
plt.scatter(x, y, color='red') # scatter plot for dataset
plt.plot(x_grid, linear_regressor_2.predict\
         (polynomial_transformer.fit_transform(x_grid)), color='blue') # prediction plot
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary ($)')
plt.show()

# predicting single results with linear and polynomial regressions
linear_reg_salary_prediction = linear_regressor.predict([[6.5]]) # predicting single result for linear regression
polynomial_reg_salary_prediction =\
    linear_regressor_2.predict(polynomial_transformer.fit_transform([[6.5]]))
print(linear_reg_salary_prediction)
print(polynomial_reg_salary_prediction)