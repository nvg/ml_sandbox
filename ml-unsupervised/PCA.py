#!/usr/bin/env python
# coding: utf-8

# # Principal Component Analysis
# 
# PCA helps you interpret your data, but it will not always find the important patterns. PCA simplifies the complexity in high-dimensional data while retaining trends and patterns. It does this by transforming the data into fewer dimensions, which act as summaries of features.
# 
# PCA reduces data by geometrically projecting them onto lower dimensions called principal components (PCs), with the goal of finding the best summary of the data using a limited number of PCs. The first PC is chosen to minimize the total distance between the data and their projection onto the PC. By minimizing this distance, we also maximize the variance of the projected points, σ2. The second (and subsequent) PCs are selected similarly, with the additional requirement that they be uncorrelated with all previous PCs. 
# 
# PCA is a good data summary when the interesting patterns increase the variance of projections onto orthogonal components. But PCA also has limitations that must be considered when interpreting the output: the underlying structure of the data must be linear (kPCA will help here), patterns that are highly correlated may be unresolved because all PCs are uncorrelated, and the goal is to maximize variance and not necessarily to find clusters
# 
# https://www.nature.com/articles/nmeth.4346

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


# In[17]:


dataset = pd.read_csv('data/Wine.csv')


# In[18]:


dataset.head()


# In[19]:


dataset.info()


# In[ ]:


X = dataset.iloc[:, 0:13].values # info
y = dataset.iloc[:, 13].values # customer segment


# In[3]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 


# In[4]:


# Scale features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 


# In[5]:


# Reduce dimentions
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_ 


# In[7]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, multi_class='auto', solver='lbfgs')
classifier.fit(X_train, y_train) 


# In[8]:


y_pred = classifier.predict(X_test)


# In[9]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 


# In[15]:


from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show() 


# In[21]:


# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show() 


# In[ ]:




