#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

2021.09.10
author: yasin sahin
written to learn data processing tools for machine learning


"""
# importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Data.csv') # importing data
x = dataset.iloc[:,:-1].values # taking independent variables
y = dataset.iloc[:,-1].values # taking dependent variable (last column)

# Replacing missing data values by taking average
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])


# Encoding categorical data for independent variable using one hot encoding
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])],\
                       remainder='passthrough')
x = np.array(ct.fit_transform(x))

# Encoding categotircal data for dependent variable using label encoder
le = LabelEncoder()
y = le.fit_transform(y)

# splitting train and test set from data
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.2,random_state=(1))
    
# scaling independent variables
sc = StandardScaler()
x_train[:,3:] = sc.fit_transform(x_train[:,3:])
x_test[:,3:] = sc.transform(x_test[:,3:])

