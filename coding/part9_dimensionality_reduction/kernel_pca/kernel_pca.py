#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2021.09.18
author: yasin sahin
written to construct kernel pca algorithm together with logistic regression

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# importing dataset
dataset = pd.read_csv('Wine.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# splitting training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Scaling independent variables
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# applying PCA and reducing dimension to two
kpca = KernelPCA(n_components = 2, kernel = 'rbf') # initiliazing pca instant
x_train = kpca.fit_transform(x_train)
x_test = kpca.transform(x_test)

# fitting logistic regression classification model to dataset
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

# predicting test set results and calculating confusion matrix and accuracy
y_pred = classifier.predict(x_test)
c_m = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy is: {accuracy}')


# visualizing training results
x_set, y_set = x_train, y_train
x_1, x_2 = np.meshgrid(np.arange(start = x_set[:,0].min() - 1, stop = x_set[:,0].max() + 1, step = 0.01), \
                       np.arange(start = x_set[:,1].min() - 1, stop = x_set[:,1].max() + 1, step = 0.01))

plt.contourf(x_1, x_2, classifier.predict(np.array([x_1.ravel(),x_2.ravel()]).T).reshape(x_1.shape), alpha = 0.75, cmap = ListedColormap(('red','green','blue')))
plt.xlim(x_1.min(),x_1.max())
plt.ylim(x_2.min(),x_2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], c = ListedColormap(('red','green','blue'))(i), label =j)
plt.title('Logistic Regression (training results)')
plt.xlabel('KPC1')
plt.ylabel('KPC2')
plt.legend()
plt.show()



# visualizing test results
x_set, y_set = x_test, y_test
x_1, x_2 = np.meshgrid(np.arange(start = x_set[:,0].min() - 1, stop = x_set[:,0].max() + 1, step = 0.01), \
                       np.arange(start = x_set[:,1].min() - 1, stop = x_set[:,1].max() + 1, step = 0.01))

plt.contourf(x_1, x_2, classifier.predict(np.array([x_1.ravel(),x_2.ravel()]).T).reshape(x_1.shape), alpha = 0.75, cmap = ListedColormap(('red','green','blue')))
plt.xlim(x_1.min(),x_1.max())
plt.ylim(x_2.min(),x_2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], c = ListedColormap(('red','green','blue'))(i), label =j)
plt.title('Logistic Regression (test results)')
plt.xlabel('KPC1')
plt.ylabel('KPC2')
plt.legend()
plt.show()
