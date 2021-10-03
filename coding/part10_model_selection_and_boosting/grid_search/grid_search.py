#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2021.09.18
author: yasin sahin
written to construct grid search model

"""

# importing necessary libraries
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# uploading dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# seperating training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# creating a pipeline to diminish data leakage due to scaling
# pipe = Pipeline([('sc', StandardScaler()),('svc', SVC(kernel = 'rbf', random_state = 0))])

# creating a pipeline to diminish data leakage due to scaling
pipe = Pipeline([
    ('sc', StandardScaler()),
    ('svc', SVC())
    ])

# determining grid parameters
parameters = [{'svc__C': [0.25, 0.5, 0.75, 1], 'svc__kernel': ['linear']},
              {'svc__C': [0.25, 0.5, 0.75, 1], 'svc__kernel': ['rbf'], 'svc__gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

# Finding the best model through GridSearch
grid_search = GridSearchCV(estimator = pipe,
                      param_grid = parameters, # grid search parameters
                      scoring = 'accuracy', # scoring according to accuracy
                      cv = 10, # cross-validation generator
                      n_jobs = -1) # using whole processor power
    
grid_search.fit(x_train, y_train) # fitting the model to training set

best_accuracy = grid_search.best_score_ # taking the best score from model
best_parameters = grid_search.best_params_ # taking the best parameters from model
print(f'Best Accuracy: {best_accuracy*100 :.2f} %]')
print(f'Best Parameters: {best_parameters}]')