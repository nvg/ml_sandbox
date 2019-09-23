#!/usr/bin/env python
# coding: utf-8

# # K-Means
# 
# K-means groups similar data points together to discover underlying patterns. For this, K-means looks for a fixed number (k) of clusters in a dataset. A cluster refers to a collection of data points aggregated together because of certain similarities. To choose the optimal number of clusters use the elbow method.
# 
# K-means works as follows. The user inputs the value of K then runs the following iterative procedure:
# 
# 1. Randomly assign a number, from 1 to K, to each of the observations. These serve as initial cluster assignments for the observations.
# 2. Iterate until the cluster assignments stop changing:
#   * For each of the K clusters, compute the cluster centroid.
#   * Assign each observation to the cluster whose centroid is closest.
# 
# ## WCSS
# 
# Within-cluster sums of squares (WCSS) is used to maximize the distance between clusters. WCSS measures how close each point is to its cluster mean by summing the squared distance from each point to its cluster center.

# In[1]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data/Mall_Customers.csv')


# In[2]:


dataset.head()


# In[3]:


dataset.info()


# In[4]:


X = dataset.iloc[:, [3, 4]].values # contains annual income and spending score
y = dataset.iloc[:, 3].values # contains annual income

# avoid scaler errors down the road
X = X.astype(float)
y = y.astype(float)


# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[6]:


## Scale Features
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))


# In[7]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[8]:


# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

