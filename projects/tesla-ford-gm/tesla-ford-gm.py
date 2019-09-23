#!/usr/bin/env python
# coding: utf-8

# # Analysis of Tesla Stock vs Ford and GM

# ### Analyze performance of Tesla Stocks, compared to GM and Ford

# In[1]:


import pandas as pd
import numpy as np
import pandas_datareader
import pandas_datareader.data as data
import datetime

# Look at a 5 year window
start = datetime.datetime(2014, 1, 1)
end = datetime.datetime(2019, 9, 1)

tesla = data.DataReader('TSLA', 'yahoo', start, end)
ford = data.DataReader('F', 'yahoo', start, end)
gm = data.DataReader('GM', 'yahoo', start, end)

tesla.head()
ford.head()
gm.head()


# In[10]:


# Look at the opening price:
#
# - The opening price is the price at which a security first trades when an exchange opens for the day.
# - An opening price is not identical to the previous day's closing price.
# - There are several day-trading strategies based on the opening price of a market or security.
#

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

tesla['Open'].plot(label='Tesla', figsize=(12, 8), title='Opening price')
gm['Open'].plot(label='GM', figsize=(12, 8), title='Opening price')
ford['Open'].plot(label='Ford', figsize=(12, 8), title='Opening price')

plt.legend();


# In[12]:


# How many shares were traded each day. A high daily volume is common when stock-specific news items are released
# or when the market moves significantly, while a low daily volume can occur on light-news days and calm days
# for the stock market.

tesla['Volume'].plot(label='Tesla', figsize=(12, 8), title='Volume')
gm['Volume'].plot(label='GM', figsize=(12, 8), title='Volume')
ford['Volume'].plot(label='Ford', figsize=(12, 8), title='Volume')

plt.legend();


# In[47]:


print("Ford: ", (ford['Volume'].idxmax()))
print("GM:   ", gm['Volume'].idxmax())
print("Tesla:", tesla['Volume'].idxmax())


# In[48]:


tesla['Total Traded'] = tesla['Open'] * tesla['Volume']
ford['Total Traded'] = ford['Open'] * ford['Volume']
gm['Total Traded'] = gm['Open'] * gm['Volume']

tesla['Total Traded'].plot(label='Tesla', figsize=(12, 8), title='Total Traded')
gm['Total Traded'].plot(label='GM', figsize=(12, 8), title='Total Traded')
ford['Total Traded'].plot(label='Ford', figsize=(12, 8), title='Total Traded')

plt.legend();


# In[50]:


print(tesla['Total Traded'].idxmax())


# In[51]:


tesla.head()


# In[24]:


# Moving averages - 
# Helps to identify the trend direction and to determine support and resistance levels. 
# While moving averages are useful enough on their own, they also form the basis for other
# technical indicators such as the moving average convergence divergence (MACD).

Because we have extensive definitions and articles around specific types of moving averages, we will only define the term "moving average" generally here.

gm['MA50'] = gm['Open'].rolling(50).mean()
gm['MA200'] = gm['Open'].rolling(200).mean()
gm[['Open', 'MA50', 'MA200']].plot(figsize=(16, 8))


# In[28]:


from pandas.plotting import scatter_matrix

car_comp = pd.concat([tesla["Open"], gm['Open'], ford['Open']], axis=1)
car_comp.columns = ['Tesla Open', 'GM Open', 'Ford Open']

scatter_matrix(car_comp, figsize=(8, 8), alpha=0.2, hist_kwds={'bins':50})


# In[30]:


# tesla['returns'] = (tesla['Close'] / tesla['Close'].shift(1)) - 1 - Same calc as below
tesla['returns'] = tesla['Close'].pct_change(1)
ford['returns'] = ford['Close'].pct_change(1)
gm['returns'] = gm['Close'].pct_change(1)


# In[31]:


ford['returns'].hist(bins=100)


# In[33]:


gm['returns'].hist(bins=100)


# In[34]:


tesla['returns'].hist(bins=100, label='Tesla')


# In[36]:


tesla['returns'].hist(bins=100, label='Tesla', figsize=(10, 8), alpha=0.4)
gm['returns'].hist(bins=100, label='GM', figsize=(10, 8), alpha=0.4)
ford['returns'].hist(bins=100, label='Ford', figsize=(10, 8), alpha=0.4)
plt.legend()


# In[37]:


tesla['returns'].plot(kind='kde', label='Tesla', figsize=(10,8))
ford['returns'].plot(kind='kde', label='Ford', figsize=(10,8))
gm['returns'].plot(kind='kde', label='GM', figsize=(10,8))
plt.legend()


# In[38]:


# What's the aggregate amount an investment has gained or lost over time, independent of the period of time involved
tesla['Cum Ret'] = (1+tesla['returns']).cumprod()
gm['Cum Ret'] = (1+gm['returns']).cumprod()
ford['Cum Ret'] = (1+ford['returns']).cumprod()


# In[52]:


tesla['Cum Ret'].plot(label='Tesla', figsize=(12, 8), title='Cum Ret')
gm['Cum Ret'].plot(label='GM', figsize=(12, 8), title='Cum Ret')
ford['Cum Ret'].plot(label='Ford', figsize=(12, 8), title='Cum Ret')

plt.legend();

