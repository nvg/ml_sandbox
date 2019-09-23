#!/usr/bin/env python
# coding: utf-8

# # Stock Market Analysis
# 
# Looking at techn stocks:
# 
# * change in price
# * daily return on average
# * moving averages
# * correlations
# * value & risk
# * behavior prediction

# In[1]:


from __future__ import division

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

from pandas_datareader import DataReader
from datetime import datetime


# In[2]:


tech_stocks = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)


# In[3]:


for stock in tech_stocks:
    globals()[stock] = DataReader(stock, 'yahoo', start, end)


# In[4]:


AAPL['Adj Close'].plot(legend=True, figsize=(10, 4))


# In[5]:


AAPL['Volume'].plot(legend=True, figsize=(10, 4))


# ## Calculate moving averages
# 
# These averages reduce noise (smooth data) by calculating averages over a certain period of time. There are two main types:
# 
# * Simple Moving Average - mean of a given set of prices
# * Exponential Moving Average - gives greater weight to more recent data
# 
# When prices cross their moving averages, it's a trading signal

# In[6]:


moving_averages = [10, 20, 50]

for ma in moving_averages:
    col_name = "MA for %s days" % (str(ma))
    
    AAPL[col_name] = AAPL['Adj Close'].rolling(ma).mean()


# In[7]:


AAPL[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(subplots=False, figsize=(10, 4))


# In[8]:


AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()
AAPL['Daily Return'].plot(figsize=(10, 4), legend=True, linestyle="--", marker='o')


# In[9]:


sns.distplot(AAPL['Daily Return'].dropna(), bins = 100, color = 'purple')


# In[10]:


AAPL['Daily Return'].hist(bins=100)


# In[11]:


closing_df = DataReader(tech_stocks, 'yahoo', start, end)['Adj Close']
tech_rets = closing_df.pct_change()


# In[12]:


sns.jointplot('GOOG', 'GOOG', tech_rets, kind='scatter', color='seagreen')


# In[13]:


sns.jointplot('GOOG','MSFT', tech_rets, kind='scatter', color='seagreen')


# In[14]:


sns.pairplot(tech_rets.dropna())


# In[15]:


returns_fig = sns.PairGrid(tech_rets.dropna())
returns_fig.map_upper(plt.scatter, color='purple')
returns_fig.map_lower(sns.kdeplot, cmap='cool_d')
returns_fig.map_diag(plt.hist, bins=30)


# In[16]:


returns_fig = sns.PairGrid(closing_df.dropna())
returns_fig.map_upper(plt.scatter, color='purple')
returns_fig.map_lower(sns.kdeplot, cmap='cool_d')
returns_fig.map_diag(plt.hist, bins=30)


# ## Risk Analysis
# 
# Risk is the standard deviation of daily returns. We compare that with the expected returns.

# In[17]:


rets = tech_rets.dropna()
area = np.pi * 20

plt.scatter(rets.mean(), rets.std(), s=area, alpha = 0.5)
plt.ylim([0.01, 0.025])
plt.xlim([-0.003, 0.004])
plt.xlabel('Expected Return')
plt.ylabel('Risk')

for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (1, 1),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-0.3'))


# Based on this graph, it might make sense to buy MSFT stocks.
# 
# **Value at risk.**
# 
# Let's try a basic Monte Carlo method.

# In[18]:


days = 365
dt = 1/days
mu = rets.mean()['GOOG']
sigma = rets.std()['GOOG']

def stock_monte_carlo(start_price, days, mu, sigma):
    price = np.zeros(days)
    price[0] = start_price
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    for x in range(1, days):
        shock[x] = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
        drift[x] = mu * dt
        price[x] = price[x-1] + (price[x-1] * (drift[x] + shock[x]))

    return price


# In[19]:


GOOG.head()


# In[20]:


start_price = 1231.15
for run in range(100):
    plt.plot(stock_monte_carlo(start_price, days, mu, sigma))
plt.xlabel("Days")
plt.ylabel("Price")
plt.title("Monte Carlo Analysis for Google")


# In[21]:


runs = 10000
simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = stock_monte_carlo(start_price, days, mu, sigma)[days - 1]


# In[22]:


q = np.percentile(simulations, 1)


# In[23]:


plt.hist(simulations, bins=200)
plt.figtext(0.6, 0.8, s="Start price $%.2f" % start_price)
plt.figtext(0.6, 0.7, s="Mean final price $%.2f" % simulations.mean())
plt.figtext(0.6, 0.6, s="VaR(0.99) $%.2f" % (start_price - q))
plt.figtext(0.15, 0.6, s="q(0.99) $%.2f" % q)
plt.axvline(x=q, linewidth=4, color='r')
plt.title(u"Final price distribution for Google Stock after %s days" % days, weight = 'bold')

