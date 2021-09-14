#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2021.09.14
author: yasin sahin
written for selecting the best classification model

"""

# importing necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_processing import DataProcessor
from sklearn.preprocessing import StandardScaler

class ClassifierContainer(DataProcessor):
    """
    This class is intend to contain classification models
    without the need of importing any other libraries.
    
    :Includes::
        
        logistic regression,  
        
        naive bayes,  
        
        SVM,
        
        decision tree, 
        
        random forest,
        
        k-nearest neighbor
    
    """
        
    def logistic(self):
        """
        
        This method creates, fits  and predicts logistic classifier model
        
        Returns
        -------
        classifier_choice : object
            object of chosen classifier.
        y_pred : float64
            predicted dependent variables from test set.
        accuracy : float64
            accuracy score of current classifier.

        """
        sc = StandardScaler() # assigning standard scaler for independent variables
        # fitting and scaling independent variable
        x_train = sc.fit_transform(self.x_train_)
        x_test = sc.transform(self.x_test_)
        classifier_choice = LogisticRegression(random_state = self.random_state_) # choosing related classifier
        classifier_choice.fit(x_train, self.y_train_) # training related classifier
        y_pred = classifier_choice.predict(x_test) # predicting results for given test set
        # outputing classifier, prediction, accuracy score of classifier
        return classifier_choice, y_pred, accuracy_score(self.y_test_,y_pred)
    def naive_bayes(self):
        """
        
        This method creates, fits  and predicts naive bayes classifier model
        
        Returns
        -------
        classifier_choice : object
            object of chosen classifier.
        y_pred : float64
            predicted dependent variables from test set.
        accuracy : float64
            accuracy score of current classifier.

        """
        sc = StandardScaler() # assigning standard scaler for independent variables
        # fitting and scaling independent variable
        x_train = sc.fit_transform(self.x_train_)
        x_test = sc.transform(self.x_test_)
        classifier_choice = GaussianNB() # choosing related classifier
        classifier_choice.fit(x_train, self.y_train_) # training related classifier
        y_pred = classifier_choice.predict(x_test) # predicting results for given test set
        # outputing classifier, prediction, accuracy score of classifier
        return classifier_choice, y_pred, accuracy_score(self.y_test_,y_pred)
    
    def svm(self):
        """
        
        This method creates, fits  and predicts svm classifier model
        
        Returns
        -------
        classifier_choice : object
            object of chosen classifier.
        y_pred : float64
            predicted dependent variables from test set.
        accuracy : float64
            accuracy score of current classifier.

        """
        sc = StandardScaler() # assigning standard scaler for independent variables
        # fitting and scaling independent variable
        x_train = sc.fit_transform(self.x_train_)
        x_test = sc.transform(self.x_test_)
        classifier_choice = SVC(kernel = self.kernel_, random_state = self.random_state_) # choosing related classifier
        classifier_choice.fit(x_train, self.y_train_) # training related classifier
        y_pred = classifier_choice.predict(x_test) # predicting results for given test set
        # outputing classifier, prediction, accuracy score of classifier
        return classifier_choice, y_pred, accuracy_score(self.y_test_,y_pred)
    def decision_tree(self):
        """
        
        This method creates, fits  and predicts decision tree classifier model
        
        Returns
        -------
        classifier_choice : object
            object of chosen classifier.
        y_pred : float64
            predicted dependent variables from test set.
        accuracy : float64
            accuracy score of current classifier.

        """
        sc = StandardScaler() # assigning standard scaler for independent variables
        # fitting and scaling independent variable
        x_train = sc.fit_transform(self.x_train_)
        x_test = sc.transform(self.x_test_)
        # choosing related classifier
        classifier_choice = DecisionTreeClassifier(criterion = self.criterion_, \
                                                   random_state = self.random_state_)
        classifier_choice.fit(x_train, self.y_train_) # training related classifier
        y_pred = classifier_choice.predict(x_test) # predicting results for given test set
        # outputing classifier, prediction, accuracy score of classifier
        return classifier_choice, y_pred, accuracy_score(self.y_test_,y_pred)
    def random_forest(self):
        """
        
        This method creates, fits  and predicts random forest classifier model
        
        Returns
        -------
        classifier_choice : object
            object of chosen classifier.
        y_pred : float64
            predicted dependent variables from test set.
        accuracy : float64
            accuracy score of current classifier.

        """
        sc = StandardScaler() # assigning standard scaler for independent variables
        # fitting and scaling independent variable
        x_train = sc.fit_transform(self.x_train_)
        x_test = sc.transform(self.x_test_)
        # choosing related classifier
        classifier_choice = RandomForestClassifier(n_estimators = self.n_estimators_,\
                                                   criterion = self.criterion_, \
                                                       random_state = self.random_state_)
        classifier_choice.fit(x_train, self.y_train_) # training related classifier
        y_pred = classifier_choice.predict(x_test) # predicting results for given test set
        # outputing classifier, prediction, accuracy score of classifier
        return classifier_choice, y_pred, accuracy_score(self.y_test_,y_pred)
    def knn(self):
        """
        
        This method creates, fits  and predicts k-nearest neighbor classifier model
        
        Returns
        -------
        classifier_choice : object
            object of chosen classifier.
        y_pred : float64
            predicted dependent variables from test set.
        accuracy : float64
            accuracy score of current classifier.

        """
        sc = StandardScaler() # assigning standard scaler for independent variables
        # fitting and scaling independent variable
        x_train = sc.fit_transform(self.x_train_)
        x_test = sc.transform(self.x_test_)
        # choosing related classifier
        classifier_choice = KNeighborsClassifier(n_neighbors = self.n_neighbors_,\
                                                 metric = self.metric_, \
                                                     p = self.p_)
        classifier_choice.fit(x_train, self.y_train_) # training related classifier
        y_pred = classifier_choice.predict(x_test) # predicting results for given test set
        # outputing classifier, prediction, accuracy score of classifier
        return classifier_choice, y_pred, accuracy_score(self.y_test_,y_pred)

