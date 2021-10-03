#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2021.09.15
author: yasin sahin
written to construct a k-means clustering model

"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# importing dataset
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:,[3,4]].values

# visualizing WCSS result to choose number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# training K-means model for 5 clusters
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(x)

# visualizing clusters
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1], s = 100, c = 'red', label = 'Cluster-1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1], s = 100, c = 'blue', label = 'Cluster-2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1], s = 100, c = 'green', label = 'Cluster-3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1], s = 100, c = 'cyan', label = 'Cluster-4')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1], s = 100, c = 'magenta', label = 'Cluster-5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],\
            s = 300, c = 'yellow', label = 'Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()