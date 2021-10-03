#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2021.09.18
author: yasin sahin
written to construct k-fold cross validation model

"""

# importing necessary libraries
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# uploading dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# seperating training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# scaling independent variables
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# creating and fitting SVM classifier
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)

# applying k-fold cross validation
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
print(f'Accuracy: {accuracies.mean()*100 :.2f} %')
print(f'Standard Deviation: {accuracies.std()*100 :.2f} %')