#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2021.09.11
author: yasin sahin
written for constructing simple linear regression model

"""
# importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# importing data
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


# splitting data into training and trest set
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size = 0.2, random_state = 0)

# training a linear regression model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# predicting the linear regression model
y_pred = regressor.predict(x_test)

# visualizing training set results
plt.scatter(x_train, y_train, color='red') # scatter plot for training set
plt.plot(x_train, regressor.predict(x_train), color='blue') # line plot for regression line
plt.title('Training Salary vs. Experience')
plt.xlabel('Experience (Years)')
plt.ylabel('Salary ($)')
plt.show()


# visualizing test set results
plt.scatter(x_test, y_test, color='red') # scatter plot for training set
plt.plot(x_train, regressor.predict(x_train), color='blue') # line plot for regression line
plt.title('Test Salary vs. Experience')
plt.xlabel('Experience (Years)')
plt.ylabel('Salary ($)')
plt.show()



