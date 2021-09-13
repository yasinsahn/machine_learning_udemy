#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

2021.09.12
author: yasin sahin
written to create data processing object for machine learning


"""
# importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split

class DataProcessor:
    """
    This class includes data preprocessing techniques.
    """
    def prepare_data_from_csv(self,filename):
        # importing the data
        dataset = pd.read_csv(filename)
        x = dataset.iloc[:,:-1].values
        y = dataset.iloc[:,-1].values
        return x, y
    def create_train_test_set(self, x, y, test_size=0.2):
        # prepparing test & training sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size)
        return x_train, x_test, y_train, y_test
    def replace_missing(self,x=[[]],y=[[]],col_idx1=0,col_idx2=1):
        if len(x)>1:
            # Replacing missing data values by taking average
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer.fit(x[:,col_idx1:col_idx2])
            x[:,col_idx1:col_idx2] = imputer.transform(x[:,col_idx1:col_idx2])
            return x
        if len(y)>1:
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer.fit(y)
            y = imputer.transform(y)
            return y
    def one_hot_encoder(self,x,col_idx=0):
        # Encoding categorical data for independent variable using one hot encoding
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [col_idx])],\
                              remainder='passthrough')
        return np.array(ct.fit_transform(x))
    def label_encoder(self,x,col_idx = 0):
        # Encoding categotircal data for dependent variable using label encoder
        le = LabelEncoder()
        x = le.fit_transform(x[:,col_idx])
        return x
    def create_polynomial(self,x,degree=4):
        # training polynomial regression model on whole data
        polynomial_transformer = PolynomialFeatures(degree=degree) # creating polynomial transformer object for specified degree
        x_poly = polynomial_transformer.fit_transform(x) # transform linear independent variable to polynomial
        return x_poly
    