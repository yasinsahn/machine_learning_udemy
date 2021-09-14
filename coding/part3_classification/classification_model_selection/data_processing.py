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
        """
        This method prepare dependent/independent variables from .csv file

        Parameters
        ----------
        filename : str
                name of .csv data file

        Returns
        -------
        x : float64
            independent variables
        y : float64
            dependent variables

        """
        # importing the data
        dataset = pd.read_csv(filename)
        x = dataset.iloc[:,:-1].values
        y = dataset.iloc[:,-1].values
        return x, y
    def create_train_test_set(self, x, y, test_size=0.2, random_state = None):
        """
        This method creates train/test set from dataset

        Parameters
        ----------
        x : float64
            independent variables
        y : float644
            dependent variables
        test_size : float, optional
            ratio of test set to the total dataset. The default is 0.2.

        Returns
        -------
        x_train : float64
            training set independent variables
        x_test : float64
            test set independent variables
        y_train : float64
            training set dependent variables
        y_test : float64
            test set dependent variables

        """
        # prepparing test & training sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = random_state)
        return x_train, x_test, y_train, y_test
    def replace_missing(self,x=[[]],y=[[]],col_idx1=0,col_idx2=1):
        """
        This method replaces missing values on dataset to mean of other values
        
        Makes replacing operation only for x or y at once
        
        Parameters
        ----------
        x : float64, optional
            independent variables of dataset. The default is [[]].
        y : float64, optional
            dependent variable of dataset. The default is [[]].
        col_idx1 : int, optional
            column start index of independent variable. The default is 0.
        col_idx2 : int, optional
            column end index of independent variable. The default is 1.

        Returns
        -------
        float64
            Filled array of dependent or independent variables.

        """
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
        """
        
        This method applies one hot encoder to independent variables.

        Parameters
        ----------
        x : float64
            independent variables of dataset.
        col_idx : int, optional
            column of independent variable to perform encoding. The default is 0.

        Returns
        -------
        float64
            encoded independent variable.

        """
        # Encoding categorical data for independent variable using one hot encoding
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [col_idx])],\
                              remainder='passthrough')
        return np.array(ct.fit_transform(x))
    def label_encoder(self,x,col_idx = 0):
        """
        This method applies label encoder to independent/dependent variables of dataset.

        Parameters
        ----------
        x : float64
            independent/dependent variable of dataset.
        col_idx : int, optional
            column of independent variable to perform encoding. The default is 0.

        Returns
        -------
        x : float64
            encoded independent variable.

        """
        # Encoding categotircal data for dependent variable using label encoder
        le = LabelEncoder()
        x = le.fit_transform(x[:,col_idx])
        return x
    def create_polynomial(self,x,degree=4):
        """
        This method creates polynomial array for polynomial linear regression

        Parameters
        ----------
        x : float64
            independent variable of dataset.
        degree : int, optional
            degree of the polynomial. The default is 4.

        Returns
        -------
        x_poly : float64
            polynomial independent variable.

        """
        # training polynomial regression model on whole data
        polynomial_transformer = PolynomialFeatures(degree=degree) # creating polynomial transformer object for specified degree
        x_poly = polynomial_transformer.fit_transform(x) # transform linear independent variable to polynomial
        return x_poly
    