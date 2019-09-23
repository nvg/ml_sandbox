#!/usr/bin/env python
# coding: utf-8

# # Linear Regression
# 
# The formula for a regression line is:
# ```
# Y = βo + β1X + ∈
# ```
# 
# where, 
# Y - Dependent variable
# X - Independent variable
# βo - Intercept
# β1 - Slope
# ∈ - Error
# 
# ### Assumptions
# 
# * Dependent var & independent var are linearly related (change in DV changes IV by a constant amount and is independent of other variables)
# * There is no correlation among independent variables (no multicollinearity)
# * The error terms has constant variance (no heteroskedestacity)
# * The error terms must be uncorrelated i.e. error at ∈t must not indicate the at error at ∈t+1 (no autocorrelation)
# * The dependent variable and the error terms must possess a normal distribution.
# 
# ### Violations
# 
# * Residual vs. Fitted Values Plot (shows heteroskedasticity) - should not show any pattern
# * Normality Q-Q plot - should show a straight line
# * Scale location plot - should not show any pattern
# 
# ### Accuracy
# 
# * Use a different model (e.g. tree-based)
# * For non-linearity, transform the IVs using sqrt, log, square, etc.
# * For heteroskedasticity, transform the DV using sqrt, log, square, or use weighted least square method
# * For multicollinearity, use a correlation matrix to check correlated variables. 
# 
# ### Assessment
# 
# * R Square (Coefficient of Determination) - increases as # of vars increase - use adjusted R^2
# * Adjusted R²
# * F Statistics - It evaluates the overall significance of the model - the ratio of explained variance by the model by unexplained variance. It compares the full model with an intercept only (no predictors) model. Its value can range between zero and any arbitrary large number. Naturally, higher the F statistics, better the model.
# * MSE / MSE / MAE - Lower the number, better the model:
#   * MSE - Mean squared error. Amplifies the impact of outliers
#   * MAE - Mean absolute error. It is robust against the effect of outliers
#   * RMSE - Root mean square error. Tells how far on an average, the residuals are from zero.
# 
# [Hackerearth.com - Intro to Linear Regression](https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-guide-regression-analysis-plot-interpretations/tutorial/)

# In[1]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_boston

boston = load_boston()


# In[3]:


print(boston.DESCR)


# In[4]:


plt.hist(boston.target, bins=50)
plt.xlabel('Prices in 1000s')
plt.ylabel('# of houses')


# In[5]:


plt.scatter(boston.data[:, 5], boston.target)
plt.ylabel('Price in 1000s')
plt.xlabel('# of rooms')


# In[6]:


boston_df = DataFrame(boston.data)
boston_df.columns = boston.feature_names


# In[7]:


boston_df.head()


# In[8]:


boston_df['Price'] = boston.target


# In[9]:


boston_df.head()


# In[10]:


sns.lmplot('RM', 'Price', data=boston_df)


# In[11]:


import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[12]:


X = boston_df.drop('Price', 1)
y = boston_df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

reg = LinearRegression()
reg.fit(X_train, y_train)


# In[13]:


print(' The estimated intercept coefficient is %.2f ' %reg.intercept_)
print(' The number of coefficients used was %d ' % len(reg.coef_))


# In[14]:


y_pred = reg.predict(X_test)


# In[15]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R^2', metrics.r2_score(y_test, y_pred))


# In[16]:


# Residual Plots
pred_train = reg.predict(X_train)
pred_test = reg.predict(X_test)

# Scatter plot the training data
train = plt.scatter(pred_train,(y_train-pred_train),c='b',alpha=0.5)

# Scatter plot the testing data
test = plt.scatter(pred_test,(y_test-pred_test),c='r',alpha=0.5)

# Plot a horizontal axis line at 0
plt.hlines(y=0,xmin=-10,xmax=50)

#Labels
plt.legend((train,test),('Training','Test'),loc='lower left')
plt.title('Residual Plots')


# In[17]:


# Residual plot of all the dataset using seaborn
sns.residplot('RM', 'Price', data = boston_df)

