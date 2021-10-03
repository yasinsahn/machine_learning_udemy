#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2021.09.18
author: yasin sahin
written to construct xg boost model

"""

# importing necessary libraries
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

# importing dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# splitting dataset into training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# training xgboost on the training set
classifier = XGBClassifier()
classifier.fit(x_train, y_train)

# predicting test set reults
y_pred = classifier.predict(x_test)
c_m = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy is: {accuracy}')

# generating cross validation score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
print(f'Accuracy: {accuracies.mean()*100 :.2f} %')
print(f'Standard Deviation: {accuracies.std()*100 :.2f} %')