#!/usr/bin/env python
# coding: utf-8

# # Support Vector Machines
# 
# SVMs are a method that uses points in a transformed problem space that best separate classes into two groups:
# * Find hyperplanes that correctly classify the training data
# * Pick the one that has the greatest distance to the points closest to it (margin)
# 
# The closest points that identifies this hyperplane are known as support vectors. The region they define around the hyperplane is known as the margin.
# 
# Data that can be separated by a line (or in general, a hyperplane) is known as linearly separable data. The hyperplane acts as a linear classifier.
# 
# SVM allows for a parameter called “C” that allows dictating the tradeoff between:
# * Having a wide margin.
# * Correctly classifying training data. A higher value of C implies you want lesser errors on the training data.
# * Smaller C gives wider margin
# 
# For non-linearly separable data, kernel trick might be used to transform the input space to a higher dimensional space: polynomial or radial basis function kernel

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
Y = iris.target

print(iris.DESCR)


# In[2]:


from sklearn.svm import SVC
model = SVC(gamma='auto') # Support Vector Classification
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
model.fit(X_train,Y_train)


# In[3]:


from sklearn import metrics

predicted = model.predict(X_test)
expected = Y_test

print(metrics.accuracy_score(expected,predicted))


# In[6]:


# Import all SVM 
from sklearn import svm

# SVC with a Linear Kernel  (our original example)
svc = svm.SVC(kernel='linear', gamma='auto', C=1.0).fit(X_train, Y_train)
predicted = svc.predict(X_test)
print("Linear: ", metrics.accuracy_score(expected,predicted))

# Gaussian Radial Bassis Function
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=1.0).fit(X_train, Y_train)
predicted = rbf_svc.predict(X_test)
print("RBF: ", metrics.accuracy_score(expected,predicted))

# SVC with 3rd degree poynomial
poly_svc = svm.SVC(kernel='poly', gamma='auto', degree=3, C=1.0).fit(X_train, Y_train)
predicted = poly_svc.predict(X_test)
print("Poly: ", metrics.accuracy_score(expected,predicted))

# SVC Linear
lin_svc = svm.LinearSVC(C=1.0, max_iter=10000).fit(X_train, Y_train)
predicted = lin_svc.predict(X_test)
print("Linear: ", metrics.accuracy_score(expected,predicted))


# In[7]:


# We'll use all the data and not bother with a split between training and testing. We'll also only use two features.
X = iris.data[:,:2]
Y = iris.target

# SVM regularization parameter
C = 1.0

# SVC with a Linear Kernel  (our original example)
svc = svm.SVC(kernel='linear', gamma='auto', C=C).fit(X, Y)

# Gaussian Radial Bassis Function
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, Y)

# SVC with 3rd degree poynomial
poly_svc = svm.SVC(kernel='poly', gamma='auto', degree=3, C=C).fit(X, Y)

# SVC Linear
lin_svc = svm.LinearSVC(C=C, max_iter=10000).fit(X,Y)


# In[8]:


## Graph the results

# Set the step size
h = 0.02

# X axis min and max
x_min=X[:, 0].min() - 1
x_max =X[:, 0].max() + 1

# Y axis min and max
y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1

# Finally, numpy can create a meshgrid
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

# title for the plots
titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']

# Use enumerate for a count
for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    plt.figure(figsize=(15,15))
    # Set the subplot position (Size = 2 by 2, position deifined by i count
    plt.subplot(2, 2, i + 1)
    
    # SUbplot spacing
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Define Z as the prediction, not the use of ravel to format the arrays
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    
    # Contour plot (filled with contourf)
    plt.contourf(xx, yy, Z, cmap=plt.cm.terrain, alpha=0.5)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Dark2)
    
    # Labels and Titles
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])
    


plt.show()


# In[ ]:




