#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2021.09.13
author: yasin sahin
written for selecting the best regression model

"""

from regression_model_container import RegressorContainer


class SelectRegressor(RegressorContainer):
    
    """
         :ONLY VALID FOR CSV DATA FILES !:
        
This class is intended to select related regression model
    
:Includes:: 
>>> Linear 
>>> Polynomial linear 
>>> SVR 
>>> Decision Tree 
>>> Random Forest
    
:Arguments::
    
>>> filename: name of the .csv data file: 
    
>>> regressor: name of regressor

>>> kernel: kernel name of SVR (only usable for SVR)

>>> n_estimators = number of trees for random forest

>>> random_state = random state
        only for random forest & decision tree
        
>>> poly_degree = degree of the polynomial
                (only for polynomial regression)

>>> test_size = split ratio for test 
                set between [0-1]

    """


    def __init__(self, filename, \
                                   regressor = 'linear',kernel='rbf', \
                                       n_estimators = 10, random_state = None,\
                                           poly_degree=4,test_size = 0.2):
        self.regressor_map_ = {
            'linear': self.linear,
            'polynomial': self.polynomial,
            'svr': self.svr,
            'decision_tree': self.decision_tree,
            'random_forest': self.random_forest
            }
        self.random_state_ = random_state
        x, y = self.prepare_data_from_csv(filename)
        self.x_train_, self.x_test_, self.y_train_, self.y_test_ = \
            self.create_train_test_set(x, y)
        self.kernel_ = kernel
        self.n_estimators_ = n_estimators
        self.poly_degree_ = poly_degree
        self.regressor_ = regressor
        self.test_size_ = test_size
    def try_regressor(self):
        """
        This method routes current regression choice to related regression method
        
        
        """
        return self.regressor_map_[self.regressor_]()



linear_regressor = SelectRegressor(filename = 'Data.csv', regressor='linear')
r_linear, y_pred_linear, r2_linear = linear_regressor.try_regressor()
print(r2_linear)


polynomial_regressor = SelectRegressor(filename='Data.csv', regressor='polynomial')
r_poly, y_pred_poly, r2_poly = polynomial_regressor.try_regressor()
print(r2_poly)


svr_regressor = SelectRegressor(filename = 'Data.csv', regressor='svr')
r_svr, y_pred_svr, r2_svr = svr_regressor.try_regressor()
print(r2_svr)

decision_tree_regressor = SelectRegressor(filename='Data.csv', regressor='decision_tree')
r_dec, y_pred_dec, r2_dec = decision_tree_regressor.try_regressor()
print(r2_dec)

random_forest_regressor = SelectRegressor(filename='Data.csv', regressor='random_forest')
r_rf, y_rf, r2_rf = random_forest_regressor.try_regressor()
print(r2_rf)
