#!/usr/bin/env python
# coding: utf-8

# # Exercise - 20 Newsgroups Dataset

# ### Introducing the assignment

# In this assignment, you will explore the **20 newsgroups text dataset**. It is one of the real-world datasets that can be directly imported from sklearn. The dataset consists of 18000 newsgroup posts on 20 topics.
# 
# The code under the following sections is implemented:
# * **Importing the necessary libraries** - **some** of the libraries necessary for the next section are imported. The rest we leave for you to import.
# * **Reading the database** - in this section, we do the following:
#     - fetch the 20 newsgroups dataset
#     - display the type of the **newsgroups** variable
#     - display the names of all classes
#     - display the first post in the database just to get an idea of how the dataset looks like
#     - display the targets
#     - using the **Counter** class, count the number of times each target has occurred in the list of targets
#     
# Your task is to create a Naive Bayes model in a similar fashion to the spam-filtering model we built during the course. Then, analyze your results with the help of a confusion matrix and a classification report. Test the performance of both the multinomial and the complement naive bayes classifiers.
# 
# *Additional task: You may try to construct the probability distribution figures of some of the classes.*
# 
# *Hint: Make use of the **categories** variable to print out the classification report.*
# 
# Good luck and have fun!

# ### Importing the necessary libraries

# In[3]:


from sklearn.datasets import fetch_20newsgroups

from collections import Counter


# In[4]:


import pandas as pd
import glob

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB,ComplementNB
from sklearn.metrics import classification_report, ConfusionMatrixDisplay,confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import numpy as np


# ### Reading the database

# In[7]:


newsgroups = fetch_20newsgroups()


# In[8]:


type(newsgroups)


# In[9]:


categories = newsgroups.target_names


# In[10]:


newsgroups.data[0]


# In[11]:


newsgroups.target


# In[12]:


Counter(newsgroups.target)


# In[13]:


inputs = newsgroups.data
target = newsgroups.target


# In[14]:


len(target)


# In[15]:


x_train,x_test,y_train,y_test=train_test_split(inputs,target,test_size=0.2,random_state=365,stratify=target)


# In[16]:


vectorizer=CountVectorizer()


# In[19]:


x_train_transf=vectorizer.fit_transform(x_train)
x_test_transf=vectorizer.transform(x_test)


# In[20]:


clf=MultinomialNB()


# In[21]:


clf.fit(x_train_transf,y_train)


# In[22]:


y_test_pred=clf.predict(x_test_transf)


# In[23]:


sns.reset_orig()
ConfusionMatrixDisplay.from_predictions(y_test,y_test_pred,
labels=clf.classes_,cmap="magma")


# In[24]:


print(classification_report(y_test,y_test_pred))


# In[ ]:




