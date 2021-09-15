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
         :ONLY VALID FOR CSV DATA FILES !:
             
This class is intend to contain classification models without the need of importing any other libraries.
    
Includes:
------------

>>> LogisticRegression 
>>> Naive Bayes 
>>> SVM 
>>> Decision Tree 
>>> Random Forest
>>> K-Nearest Neighbor

Arguments:
-------------

>>> filename (str): name of the .csv data file: 
    
>>> kernel (str, optional, defaul:rbf): 
            kernel name of SVM

>>> n_estimators (int, optional, default:10):
                number of trees for random forest

>>> random_state (int, optional, default:None):
                random state of the classifier

>>> test_size (float, optional, default:0.2):
            split ratio for test set between [0-1]
                
>>> criterion (str, optinal, default:entropy):
            tree calculating criterion 
            (decision tree & random forest)      
                
>>> n_neigbors (int, optional, default:5):
        number of neighbors (knn)
                        
>>> metric (str, optional, default:minkowski):
            distance metric (knn)
                        
>>> p (int, optional, default:2):
        distance metric degree (knn)                

Attributes:
--------------

logistic
-------
This method creates, fits  and predicts logistic classifier model

:Returns::

classifier_choice : object
    object of chosen classifier.
y_pred : float64
    predicted dependent variables from test set.
accuracy : float64
    accuracy score of current classifier.
        
naive_bayes
------------
This method creates, fits  and predicts naive bayes classifier model

:Returns::

classifier_choice : object
    object of chosen classifier.
y_pred : float64
    predicted dependent variables from test set.
accuracy : float64
    accuracy score of current classifier.
    
SVM
----------
This method creates, fits  and predicts svm classifier model

:Returns::

classifier_choice : object
    object of chosen classifier.
y_pred : float64
    predicted dependent variables from test set.
accuracy : float64
    accuracy score of current classifier.

rbf is used as default kernel
 
kernel can be determined according to sklearn.svm.SVC

decision_tree
---------------  
This method creates, fits  and predicts decision tree classifier model

:Returns::

classifier_choice : object
    object of chosen classifier.
y_pred : float64
    predicted dependent variables from test set.
accuracy : float64
    accuracy score of current classifier.

random_forest
-----------------
This method creates, fits  and predicts random forest classifier model

:Returns::

classifier_choice : object
    object of chosen classifier.
y_pred : float64
    predicted dependent variables from test set.
accuracy : float64
    accuracy score of current classifier.

knn
--------
This method creates, fits  and predicts k-nearest neighbor classifier model

:Returns::

classifier_choice : object
    object of chosen classifier.
y_pred : float64
    predicted dependent variables from test set.
accuracy : float64
    accuracy score of current classifier.


    """
    def __init__(self, filename, kernel='rbf', \
                                       n_estimators = 10, random_state = None,\
                                           test_size = 0.2, criterion = 'entropy',\
                                               n_neighbors = 5, metric = 'minkowski',p = 2):
        self.random_state_ = random_state
        x, y = self.prepare_data_from_csv(filename)
        self.test_size_ = test_size
        
        self.x_train_, self.x_test_, self.y_train_, self.y_test_ = \
            self.create_train_test_set(x, y, test_size = self.test_size_,\
                                       random_state = self.random_state_)
        self.kernel_ = kernel
        self.n_estimators_ = n_estimators
        self.criterion_ = criterion
        self.n_neighbors_ = n_neighbors
        self.metric_ = metric
        self.p_ = p
        
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
