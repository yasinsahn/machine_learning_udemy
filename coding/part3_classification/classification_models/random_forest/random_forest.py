#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2021.09.14
author: yasin sahin
written to construct a random forest classification model

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# importing dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


# splitting training and test set from dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, \
                                                    test_size=0.25, random_state = 0)

# applying standard scaling for training set    
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test= sc_x.transform(x_test)
# training logistic regression model
classifier = RandomForestClassifier(n_estimators=100, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

# predicting test set results
y_pred = classifier.predict(x_test)

# creating the confusion matrix
c_m = confusion_matrix(y_test, y_pred)
print(c_m)
print(accuracy_score(y_test, y_pred))

# visualizing training results
x_set, y_set = sc_x.inverse_transform(x_train), y_train
x_1, x_2 = np.meshgrid(np.arange(start = x_set[:,0].min() - 10, stop = x_set[:,0].max() + 10, step = 2), \
                       np.arange(start = x_set[:,1].min() - 1000, stop = x_set[:,1].max() + 1000, step = 2))

plt.contourf(x_1, x_2, classifier.predict(sc_x.transform(np.array([x_1.ravel(),x_2.ravel()]).T)).reshape(x_1.shape), alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(x_1.min(),x_1.max())
plt.ylim(x_2.min(),x_2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], c = ListedColormap(('red','green'))(i), label =j)
plt.title('Random Forest Classifier (training results)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# visualizing test results
x_set, y_set = sc_x.inverse_transform(x_test), y_test
x_1, x_2 = np.meshgrid(np.arange(start = x_set[:,0].min() - 10, stop = x_set[:,0].max() + 10, step = 2), \
                       np.arange(start = x_set[:,1].min() - 1000, stop = x_set[:,1].max() + 1000, step = 2))

plt.contourf(x_1, x_2, classifier.predict(sc_x.transform(np.array([x_1.ravel(),x_2.ravel()]).T)).reshape(x_1.shape), alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(x_1.min(),x_1.max())
plt.ylim(x_2.min(),x_2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], c = ListedColormap(('red','green'))(i), label =j)
plt.title('Random Forest Classifier (test results)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()