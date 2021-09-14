#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2021.09.12
author: yasin sahin
written for selecting the best regression model

"""

# importing necessary libraries
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from data_processing import DataProcessor
from sklearn.preprocessing import StandardScaler

class RegressorContainer(DataProcessor):
    """
    This class is intend to contain regression models
    without the need of importing any other libraries.
    
    :Includes::
        
        linear,  
        
        polynomial linear,  
        
        svr,
        
        decision tree, 
        
        random forest
    
    """
        
    def linear(self):
        """
        
        This method creates, fits  and predicts linear regression model
        
        Returns
        -------
        regressor_choice : object
            object of chosen regressor.
        y_pred : float64
            predicted dependent variables from test set.
        r2_score : float64
            r2 score of current regressor.

        """
        regressor_choice = LinearRegression() # choosing related regression
        regressor_choice.fit(self.x_train_, self.y_train_) # training related regression
        y_pred = regressor_choice.predict(self.x_test_) # predicting results for given test set
        return regressor_choice, y_pred, r2_score(self.y_test_,y_pred) # outputing regressor, prediction, r2 score of regression
    def polynomial(self):
        """
        This method creates, fits and predicts polynomial linear regression model
        
        Returns
        -------
        regressor_choice : object
            object of chosen regressor.
        y_pred : float64
            predicted dependent variables from test set.
        r2_score : float64
            r2 score of current regressor.
        
        """
        regressor_choice = LinearRegression() # choosing related regression
        x_poly = self.create_polynomial(self.x_train_,self.poly_degree_) # converting training variables to polynomial
        regressor_choice.fit(x_poly, self.y_train_) # training related regression
        # predicting results for given test set
        y_pred = regressor_choice.predict(self.create_polynomial(self.x_test_,self.poly_degree_))
        # outputing regressor, prediction, r2 score of regression
        return regressor_choice, y_pred, r2_score(self.y_test_,y_pred)
    def svr(self):
        """
        This method creates, fits, and predicts support vector regression model
        
        rbf is used as default kernel, 
        
        kernel can be determined according to sklearn.svm.SVR
        
        Uses standard scaler to scale variables
        
        Returns
        -------
        regressor_choice : object
            object of chosen regressor.
        y_pred : float64
            predicted dependent variables from test set.
        r2_score : float64
            r2 score of current regressor.
        
        """
        regressor_choice = SVR(kernel=self.kernel_) # choosing related regression
        sc_x = StandardScaler() # assigning standard scaler for independent variables
        sc_y = StandardScaler() # assigning standard scaler for dependent variables
        # fitting and scaling independent variable
        x_sr = sc_x.fit_transform(self.x_train_)
        # fitting and scaling dependent variable
        y_sr = sc_y.fit_transform(self.y_train_.reshape(len(self.y_train_),1))
        regressor_choice.fit(x_sr, y_sr) # training related regressor
        # predicting results for given test set
        y_pred = regressor_choice.predict(sc_x.transform(self.x_test_))
        # outputing regressor, prediction, r2 score of regression
        return regressor_choice, sc_y.inverse_transform(y_pred), r2_score(self.y_test_,sc_y.inverse_transform(y_pred))
    def decision_tree(self):
        """
        This method creates, fits, and predicts decision tree regression model
        
        random state of the regressor can be determined
        
        Returns
        -------
        regressor_choice : object
            object of chosen regressor.
        y_pred : float64
            predicted dependent variables from test set.
        r2_score : float64
            r2 score of current regressor.
        
        """
        # choosing related regression
        regressor_choice = DecisionTreeRegressor(random_state = self.random_state_)
        # training related regressor
        regressor_choice.fit(self.x_train_, self.y_train_)
        # predicting results for given test set
        y_pred = regressor_choice.predict(self.x_test_)
        # outputing regressor, prediction, r2 score of regression
        return regressor_choice, y_pred, r2_score(self.y_test_,y_pred)
    def random_forest(self):
        """
        This method creates, fits, and predicts random forest regression model
        
        10 is used as default number of estimators
        
        random state and number of estimators of the regressor can be determined
        
        Returns
        -------
        regressor_choice : object
            object of chosen regressor.
        y_pred : float64
            predicted dependent variables from test set.
        r2_score : float64
            r2 score of current regressor.
        
        """
        # choosing related regression
        regressor_choice = RandomForestRegressor(n_estimators = self.n_estimators_, \
                                                 random_state = self.random_state_)
        # training related regressor
        regressor_choice.fit(self.x_train_, self.y_train_)  
        # predicting results for given test set
        y_pred = regressor_choice.predict(self.x_test_)
        # outputing regressor, prediction, r2 score of regression
        return regressor_choice, y_pred, r2_score(self.y_test_,y_pred)
