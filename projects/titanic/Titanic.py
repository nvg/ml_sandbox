#!/usr/bin/env python
# coding: utf-8

# # Titanic Dataset
# 
# Two main themes:
# 1. Who were the passengers?
# 2. What helped them to survive?

# In[1]:


# "Default" Imports
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Investigate Data

# In[2]:


df = pd.read_csv('data/train.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


# who were the passengers


# In[6]:


# breakdown of passenger by sex and class
sns.catplot('Pclass', data=df, kind='count', hue='Sex')


# In[7]:


def classify_passenger_as_male_female_child(passenger):
    age, sex = passenger
    
    if age < 16:
        return 'child_' + sex
    
    return sex


# In[8]:


df['person'] = df[['Age', 'Sex']].apply(classify_passenger_as_male_female_child, axis=1)


# In[9]:


sns.catplot('Pclass', data=df, kind='count', hue='person')


# Looks like there were way more children in 3rd class!

# In[10]:


df['Age'].hist(bins=70)


# In[11]:


df['Age'].mean() # What was the mean age of the passenger?


# In[12]:


df['person'].value_counts() # How many passengers of each type?


# In[13]:


fig = sns.FacetGrid(df, hue='Sex', aspect=4)
fig.map(sns.kdeplot, 'Age', shade=True)
oldest = df['Age'].max()
fig.set(xlim=(0, oldest))
fig.add_legend()


# In[14]:


fig = sns.FacetGrid(df, hue='person', aspect=4)
fig.map(sns.kdeplot, 'Age', shade=True)
oldest = df['Age'].max()
fig.set(xlim=(0, oldest))
fig.add_legend()


# Now, let's examine how many passengers of each class were there

# In[15]:


deck = df['Cabin'].dropna()


# In[16]:


deck.head()


# In[17]:


levels = []
for level in deck:
    levels.append(level[0])
    
cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']


# In[18]:


sns.catplot('Cabin', data=cabin_df, kind='count', palette='winter_d')


# In[19]:


# looks like there is a redundant T cabin
cabin_df = cabin_df[cabin_df.Cabin != 'T']


# In[20]:


sns.catplot('Cabin', data=cabin_df, kind='count', palette='summer', order=sorted(cabin_df['Cabin'].unique()))


# Where did passengers come from?

# In[21]:


sns.catplot(kind='count', x='Embarked', data=df, hue='Pclass', order=['C', 'Q', 'S'])


# Interesting fact that Queenstown passengers were mostly 3rd class. Now let's see who was alone and who was with family.

# In[22]:


df['Alone'] = df['SibSp'] + df['Parch']
df.head()


# In[23]:


df['Alone'].loc[df['Alone'] > 0] = 'With Family'
df['Alone'].loc[df['Alone'] == 0] = 'Alone'


# In[24]:


sns.catplot(kind='count', x='Alone', data=df, palette='Blues')


# Mostly people were travelling alone. So now we can move ahead to figure out the 2nd theme.
# 
# ## Survival Factors

# In[25]:


df['Survivor'] = df['Survived'].map({0: 'no', 1: 'yes'})


# In[26]:


sns.catplot(kind='count', x='Survivor', data=df, palette='Set1')


# In[27]:


#sns.catplot('Pclass', 'Survivor', data=df)
sns.factorplot(x='Pclass', y='Survived', data=df, hue='person')


# In[28]:


sns.lmplot('Age', 'Survived', data=df)


# In[29]:


sns.lmplot('Age', 'Survived', data=df, hue='Pclass', palette='winter')


# In[30]:


generations = [10, 20, 40, 60, 80]

sns.lmplot('Age', 'Survived', hue='Pclass', data=df, x_bins = generations, palette = 'winter')


# In[31]:


sns.lmplot('Age', 'Survived', hue='Sex', data=df, x_bins = generations, palette = 'winter')


# In[32]:


ds_df = df[['Cabin', 'Alone', 'Survivor']].dropna()

temp_data = []
for index, row in ds_df.iterrows():
    cabin_code = row[0][0]    
    if (cabin_code != 'T'):
        temp_data.append([cabin_code, row[1], row[2]])
        
ds_df = DataFrame(temp_data)
ds_df.columns = ['Cabin', 'Alone', 'Survivor']


# In[33]:


sns.catplot('Cabin', data=ds_df, hue='Survivor', kind='count', palette='Set1', order=sorted(cabin_df['Cabin'].unique()))


# In[34]:


sns.catplot('Alone', data=df, hue='Survivor', kind='count')

