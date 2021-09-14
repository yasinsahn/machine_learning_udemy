#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2021.09.14
author: yasin sahin
written for selecting the best classification model

"""

from classification_model_container import ClassifierContainer


class SelectClassifier(ClassifierContainer):
    """
         :ONLY VALID FOR CSV DATA FILES !:
        
This class is intended to select related classification model
    
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
    
>>> classifier (str): name of classifier

>>> kernel (str): kernel name of SVR 
                    (only usable for SVR)

>>> n_estimators (int) = number of trees 
                                for random forest

>>> random_state (int) = random state of the classifier

>>> test_size (float)= split ratio for test 
                set between [0-1]
                
>>> criterion (str) = tree calculating criterion
                        (valid for decision tree
                                 & random forest)      
                
>>> n_neigbors (int) = number of neighbors
                        (only usable for knn)
                        
>>> metric (str) = distance metric
                        (only usable for knn)
                        
>>> p (int) = distance metric degree
                        (only usable for knn)                

Returns:
-----------

>>> classifier properties (classifier object)
>>> prediction results for test set (float64)
>>> accuracy score (float64)

Attributes:
--------------

:try_classifier::

This method routes current classification choice to related classification method
        
:Returns:: 
>>> Selected classifier from classifier map

:logistic::

This method creates, fits  and predicts linear classification model
        
:naive_bayes::

This method creates, fits  and predicts naive bayes classifier model
    
:SVM::

This method creates, fits  and predicts svm classifier model

rbf is used as default kernel
 
kernel can be determined according to sklearn.svm.SVC

:decision_tree::
        
This method creates, fits  and predicts decision tree classifier model

:random_forest::

This method creates, fits  and predicts random forest classifier model

:knn::

This method creates, fits  and predicts k-nearest neighbor classifier model


    """
    def __init__(self, filename, \
                                   classifier = 'linear',kernel='rbf', \
                                       n_estimators = 10, random_state = None,\
                                           test_size = 0.2, criterion = 'entropy',\
                                               n_neighbors = 5, metric = 'minkowski',p = 2):
        self.classifier_map_ = {
            'logistic': self.logistic,
            'naive_bayes': self.naive_bayes,
            'svm': self.svm,
            'decision_tree': self.decision_tree,
            'random_forest': self.random_forest,
            'knn': self.knn
            }
        self.random_state_ = random_state
        x, y = self.prepare_data_from_csv(filename)
        self.test_size_ = test_size
        
        self.x_train_, self.x_test_, self.y_train_, self.y_test_ = \
            self.create_train_test_set(x, y, test_size = self.test_size_,\
                                       random_state = self.random_state_)
        self.kernel_ = kernel
        self.n_estimators_ = n_estimators
        self.classifier_ = classifier
        self.criterion_ = criterion
        self.n_neighbors_ = n_neighbors
        self.metric_ = metric
        self.p_ = p
    def try_classifier(self):
        """
        
        This method routes current classifier choice to related classification method

        """

        return self.classifier_map_[self.classifier_]()



logistic_classifier = SelectClassifier(filename = 'Data.csv', classifier='logistic',\
                                       test_size = 0.25, random_state = 0)
c_logistic, y_pred_logistic, accuracy_logistic = logistic_classifier.try_classifier()
print(accuracy_logistic)


knn_classifier = SelectClassifier(filename = 'Data.csv', classifier='knn',\
                                       test_size = 0.25, random_state = 0, n_neighbors = 5,\
                                           metric='minkowski', p=2)
c_knn, y_pred_knn, accuracy_knn = knn_classifier.try_classifier()
print(accuracy_knn)

svm_classifier = SelectClassifier(filename = 'Data.csv', classifier='svm',\
                                       test_size = 0.25, random_state = 0, kernel = 'rbf')
c_svm, y_pred_svm, accuracy_svm = svm_classifier.try_classifier()
print(accuracy_svm)


naive_bayes_classifier = SelectClassifier(filename = 'Data.csv', classifier='naive_bayes',\
                                       test_size = 0.25, random_state = 0)
c_nb, y_pred_nb, accuracy_nb = naive_bayes_classifier.try_classifier()
print(accuracy_nb)


decision_tree_classifier = SelectClassifier(filename = 'Data.csv', classifier='decision_tree',\
                                       test_size = 0.25, random_state = 0, criterion = 'entropy')
c_dt, y_pred_dt, accuracy_dt = decision_tree_classifier.try_classifier()
print(accuracy_dt)


random_forest_classifier = SelectClassifier(filename = 'Data.csv', classifier='random_forest',\
                                       test_size = 0.25, random_state = 0, \
                                           n_estimators = 10, criterion = 'entropy')
c_rf, y_pred_rf, accuracy_rf = random_forest_classifier.try_classifier()
print(accuracy_rf)