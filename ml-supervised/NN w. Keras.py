#!/usr/bin/env python
# coding: utf-8

# # Artificial Neural Networks
# 
# Neurons are modeled as:
# 
# ```
# y = g(w0 + sum(wi * xi))
# ```
# 
# Where w0 is the bias term, wi is the input feature, xi is the weight and g is the activation function. If no activation function is used, the model become a regression.
# 
# In a ANN, weights are adjusted based during backpropagation. One backpropagation algorithm is gradient descent.
# 
# ## Types of ANNs
# 
# Multilayer Perceptrons, or MLPs for short, are the classical type of neural network. They are comprised of one or more layers of neurons. Data is fed to the input layer, there may be one or more hidden layers providing levels of abstraction, and predictions are made on the output layer, also called the visible layer. Try them on image, text, time series, etc.
# 
# Convolutional Neural Networks (CNN) is one of the variants of neural networks used heavily in the field of Computer Vision. It derives its name from the type of hidden layers it consists of. The hidden layers of a CNN typically consist of convolutional layers, pooling layers, fully connected layers, and normalization layers. Here it simply means that instead of using the normal activation functions, convolution and pooling functions are used as activation functions.
# 
# Recurrent neural network (RNN), unlike a feedforward neural network, is a variant of a recursive artificial neural network in which connections between neurons make a directed cycle. It means that output depends not only on the present inputs but also on the previous step’s neuron state. This memory lets users solve NLP problems 
# 
# https://medium.com/@datamonsters/artificial-neural-networks-for-natural-language-processing-part-1-64ca9ebfa3b2
# 
# https://machinelearningmastery.com/when-to-use-mlp-cnn-and-rnn-neural-networks/

# In[1]:


from sklearn.datasets import load_wine
wine_data = load_wine()


# In[2]:


type(wine_data)


# In[3]:


wine_data.keys()


# In[4]:


print(wine_data['DESCR'])


# In[5]:


feat_data = wine_data['data']
labels = wine_data['target']


# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feat_data,
                                                    labels,
                                                    test_size=0.3,
                                                    random_state=101)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_x_train = scaler.fit_transform(X_train)
scaled_x_test = scaler.transform(X_test)


# In[7]:


import tensorflow as tf
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import losses, optimizers, metrics, activations

seq_model = models.Sequential()
seq_model.add(layers.Dense(units=13, input_dim=13,activation='relu'))
seq_model.add(layers.Dense(units=13, activation='relu'))
seq_model.add(layers.Dense(units=13, activation='relu'))
seq_model.add(layers.Dense(units=13, activation='relu'))
seq_model.add(layers.Dense(units=13, activation='relu'))
seq_model.add(layers.Dense(units=13, activation='relu'))
seq_model.add(layers.Dense(units=3, activation='softmax'))


# In[8]:


# seq_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# seq_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

losses.sparse_categorical_crossentropy

seq_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
seq_model.fit(scaled_x_train, y_train, epochs=500)


# In[9]:


from sklearn.metrics import classification_report

predictions = seq_model.predict_classes(scaled_x_test)
print(classification_report(predictions, y_test))

