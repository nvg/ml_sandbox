#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Capital Asset Pricing Model

from scipy import stats
import pandas as pd
import pandas_datareader as web
spy_etf = web.DataReader('SPY','yahoo')


# In[2]:


spy_etf.head()


# In[3]:


start = pd.to_datetime('2010-01-04')
end = pd.to_datetime('2019-09-13')

aapl = web.DataReader('AAPL','yahoo',start,end)


# In[4]:


aapl.head()

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

aapl['Close'].plot(label='AAPL',figsize=(10,8))
spy_etf['Close'].plot(label='SPY Index')
plt.legend()


# In[5]:


aapl['Cumulative'] = aapl['Close']/aapl['Close'].iloc[0]
spy_etf['Cumulative'] = spy_etf['Close']/spy_etf['Close'].iloc[0]


# In[6]:


aapl['Cumulative'].plot(label='AAPL',figsize=(10,8))
spy_etf['Cumulative'].plot(label='SPY Index')
plt.legend()
plt.title('Cumulative Return')


# In[7]:


aapl['Daily Return'] = aapl['Close'].pct_change(1)
spy_etf['Daily Return'] = spy_etf['Close'].pct_change(1)
plt.scatter(aapl['Daily Return'],spy_etf['Daily Return'],alpha=0.3)


# In[8]:


aapl['Daily Return'].hist(bins=100)
spy_etf['Daily Return'].hist(bins=100)

beta,alpha,r_value,p_value,std_err = stats.linregress(aapl['Daily Return'].iloc[1:],spy_etf['Daily Return'].iloc[1:])


# In[9]:


print(beta)
print(alpha)


# In[10]:


import numpy as np
noise = np.random.normal(0,0.001,len(spy_etf['Daily Return'].iloc[1:]))
noise
spy_etf['Daily Return'].iloc[1:] + noise
beta,alpha,r_value,p_value,std_err = stats.linregress(spy_etf['Daily Return'].iloc[1:]+noise,spy_etf['Daily Return'].iloc[1:])
print(beta)
print(alpha)

