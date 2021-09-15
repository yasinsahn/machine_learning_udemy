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
        
>>> classifier (str, optional, default: logistic):
                name of classifier

Returns:
-----------

>>> classifier properties (classifier object)
>>> prediction results for test set (float64)
>>> accuracy score (float64)

Attributes:
--------------

try_classifier
-----------------
This method routes current classification choice to related classification method
        
:Returns:: 
>>> Selected classifier from classifier map

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
    def __init__(self, filename, kernel='rbf', n_estimators = 10, random_state = None,\
                                           test_size = 0.2, criterion = 'entropy',\
                                               n_neighbors = 5, metric = 'minkowski',p = 2,\
                                                   classifier = 'logistic',):
        
        super().__init__(filename, kernel, n_estimators, random_state, \
                          test_size, criterion, n_neighbors, metric, p)
        self.classifier_ = classifier
        self.classifier_map_ = {
            'logistic': self.logistic,
            'naive_bayes': self.naive_bayes,
            'svm': self.svm,
            'decision_tree': self.decision_tree,
            'random_forest': self.random_forest,
            'knn': self.knn
            }

    def try_classifier(self):
        """
        
        This method routes current classifier choice to related classification method

        """

        return self.classifier_map_[self.classifier_]()



logistic_classifier = SelectClassifier(filename = 'Data.csv', classifier='logistic',\
                                       test_size = 0.25, random_state = 0)
c_logistic, y_pred_logistic, accuracy_logistic = logistic_classifier.try_classifier()
print(f'Logistic regression accuracy is: {accuracy_logistic}')


knn_classifier = SelectClassifier(filename = 'Data.csv', classifier='knn',\
                                       test_size = 0.25, random_state = 0, n_neighbors = 5,\
                                           metric='minkowski', p=2)
c_knn, y_pred_knn, accuracy_knn = knn_classifier.try_classifier()
print(f'K-nearest neighbor accuracy is: {accuracy_knn}')

svm_classifier = SelectClassifier(filename = 'Data.csv', classifier='svm',\
                                       test_size = 0.25, random_state = 0, kernel = 'rbf')
c_svm, y_pred_svm, accuracy_svm = svm_classifier.try_classifier()
print(f'SVM accuracy is: {accuracy_svm}')


naive_bayes_classifier = SelectClassifier(filename = 'Data.csv', classifier='naive_bayes',\
                                       test_size = 0.25, random_state = 0)
c_nb, y_pred_nb, accuracy_nb = naive_bayes_classifier.try_classifier()
print(f'Naive Bayes accuracy is: {accuracy_nb}')


decision_tree_classifier = SelectClassifier(filename = 'Data.csv', classifier='decision_tree',\
                                       test_size = 0.25, random_state = 0, criterion = 'entropy')
c_dt, y_pred_dt, accuracy_dt = decision_tree_classifier.try_classifier()
print(f'Decision tree accuracy is: {accuracy_dt}')


random_forest_classifier = SelectClassifier(filename = 'Data.csv', classifier='random_forest',\
                                       test_size = 0.25, random_state = 0, \
                                           n_estimators = 10, criterion = 'entropy')
c_rf, y_pred_rf, accuracy_rf = random_forest_classifier.try_classifier()
print(f'Random forest accuracy is: {accuracy_rf}')
