#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


aapl = pd.read_csv('csv/AAPL_CLOSE', index_col='Date', parse_dates=True)
cisco = pd.read_csv('csv/CISCO_CLOSE', index_col='Date', parse_dates=True)
ibm = pd.read_csv('csv/IBM_CLOSE', index_col='Date', parse_dates=True)
amzn = pd.read_csv('csv/AMZN_CLOSE', index_col='Date', parse_dates=True)


# In[3]:


stocks = pd.concat([aapl, cisco, ibm, amzn], axis = 1)
stocks.columns = ['aapl', 'cisco', 'ibm', 'amzn']
stocks.pct_change(1).mean()


# In[5]:


stocks.pct_change(1).corr()


# In[6]:


log_ret = np.log(stocks/stocks.shift(1))
log_ret.head()


# In[13]:


np.random.seed(101)

print(stocks.columns)
weights  = np.array(np.random.random(4))

print("Random weights")
print(weights)

print("Rebalance")
weights = weights / np.sum(weights)
print(weights)

# Expected return
print('Expected Portfolio Return')
exp_ret = np.sum((log_ret.mean() * weights) * 252)
print(exp_ret)

# Expected volatility
print("Expected volatility")
exp_vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
print(exp_vol)

# Sharpe Ratio
print("Sharpe Ratio")
SR = exp_ret / exp_vol
print(SR)


# In[29]:


np.random.seed(101)

num_ports = 20000
all_weights = np.zeros((num_ports, len(stocks.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for ind in range(num_ports):
    weights = np.array(np.random.random(4))
    weights = weights / np.sum(weights)
    all_weights[ind, :] = weights
    ret_arr[ind] = np.sum((log_ret.mean() * weights)* 252)    
    vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
    sharpe_arr[ind] = ret_arr[ind] / vol_arr[ind]


# In[30]:


sharpe_arr.max()


# In[31]:


sharpe_arr.argmax()


# In[26]:


all_weights[1420, :]


# In[34]:


plt.figure(figsize=(12,8))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='plasma')
plt.colorbar(label='SR')
plt.xlabel('Volatility')
plt.ylabel('Return')

max_sr_ret = ret_arr[sharpe_arr.argmax()]
max_sr_vol = vol_arr[sharpe_arr.argmax()]
plt.scatter(max_sr_vol, max_sr_ret, c='red', s=50, edgecolor='black')


# In[39]:


## Mathematical Optimization
def get_ret_vol_sr(weights):
    weights = np.array(weights)
    ret = np.sum(log_ret.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
    sr = ret/vol
    return np.array([ret,vol,sr])

from scipy.optimize import minimize

def neg_sharpe(weights):
    return  get_ret_vol_sr(weights)[2] * -1

def check_sum(weights):
    return np.sum(weights) - 1

cons = ({'type':'eq','fun': check_sum})
bounds = ((0, 1), (0, 1), (0, 1), (0, 1))
init_guess = [0.25,0.25,0.25,0.25]

opt_results = minimize(neg_sharpe,init_guess,method='SLSQP',bounds=bounds,constraints=cons)


print(opt_results)
print(opt_results.x)

get_ret_vol_sr(opt_results.x)


# In[42]:


frontier_y = np.linspace(0,0.3,100)

def minimize_volatility(weights):
    return  get_ret_vol_sr(weights)[1] 

frontier_volatility = []

for possible_return in frontier_y:
    cons = ({'type':'eq','fun': check_sum},
            {'type':'eq','fun': lambda w: get_ret_vol_sr(w)[0] - possible_return})
    result = minimize(minimize_volatility,init_guess,method='SLSQP',bounds=bounds,constraints=cons)
    frontier_volatility.append(result['fun'])

plt.figure(figsize=(12,8))
plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')

plt.plot(frontier_volatility,frontier_y,'g--',linewidth=3)


# In[ ]:




