#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes
# 
# Uses Bayes' Theorem for estimating probabilites `P(A|B) = [P(B|A) * P(A)] / P(B)`
# 
# The 'naive' is due to the assumption that is required for Bayes to work optimally - all features must be independent of each other
# 
# ### Advantages
# 
# * Easy and fast to predict the class of the test data set.
# * Performs well in multi-class prediction.
# * When assumption of independence holds, it performs better compare to other models like logistic regression and you need less training data.
# * Perform well in case of categorical input variables compared to numerical variable(s). 
# * For numerical variable, normal distribution is assumed (bell curve, which is a strong assumption).
# 
# ### Disadvantage
# 
# * If categorical variable has a category (in test data set), which was not observed in training data set, then model will assign a 0 (zero) probability and will be unable to make a prediction. This is often known as Zero Frequency. To solve this, we can use the smoothing technique. One of the simplest smoothing techniques is called Laplace estimation.
# * Naive Bayes is also known as a bad estimator, so the probability outputs are not to be taken too seriously.
# * Another limitation of Naive Bayes is the assumption of independent predictors. In real life, it is almost impossible that we get a set of predictors which are completely independent.
# 
# [Medium](https://towardsdatascience.com/all-about-naive-bayes-8e13cef044cf)
# 

# In[1]:


import pandas as pd
from pandas import Series,DataFrame

import matplotlib.pyplot as plt
import seaborn as sns

# Gaussian Naive Bayes
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


# In[2]:


iris = datasets.load_iris()

X = iris.data
Y = iris.target

model = GaussianNB()
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
model.fit(X_train,Y_train)

predicted = model.predict(X_test)
expected = Y_test
print (metrics.accuracy_score(expected, predicted))

