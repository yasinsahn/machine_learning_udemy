#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2021.09.15
author: yasin sahin
written to construct a hierarhical clustering model

"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch


# importing dataset
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:,[3,4]].values

# plotting dendrogram plot to find optimal number of clusters
dendogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Points')
plt.ylabel('Eucladian Distance')
plt.show()

hierarchical = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hierarchial = hierarchical.fit_predict(x)

# visualizing clusters
plt.scatter(x[y_hierarchial==0,0],x[y_hierarchial==0,1], s = 100, c = 'red', label = 'Cluster-1')
plt.scatter(x[y_hierarchial==1,0],x[y_hierarchial==1,1], s = 100, c = 'blue', label = 'Cluster-2')
plt.scatter(x[y_hierarchial==2,0],x[y_hierarchial==2,1], s = 100, c = 'green', label = 'Cluster-3')
plt.scatter(x[y_hierarchial==3,0],x[y_hierarchial==3,1], s = 100, c = 'cyan', label = 'Cluster-4')
plt.scatter(x[y_hierarchial==4,0],x[y_hierarchial==4,1], s = 100, c = 'magenta', label = 'Cluster-5')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()