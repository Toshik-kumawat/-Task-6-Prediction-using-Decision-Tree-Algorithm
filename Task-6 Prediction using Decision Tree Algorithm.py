#!/usr/bin/env python
# coding: utf-8

# # Task-6 Prediction using Decision Tree Algorithm
# 
# Coder - TOSHIK KUMAWAT
# 
# # Problem Statement:
# 
# ● Create the Decision Tree classifier and visualize it graphically.
# 
# ● The purpose is if we feed any new data to this classifier, it would be able to predict the right class accordingly
# 
# Dataset : https://bit.ly/3kXTdox
# 
# Task completed during Data Science & Analytics Internship @ The Sparks Foundation
# 
# # Importing all the required libraries

# In[2]:



import numpy as np
import pandas as pd
import sklearn.metrics as sm
import seaborn as sns
import matplotlib.pyplot as mt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report


# # Reading the data

# In[5]:


data=pd.read_csv('C://Users//toshi//Downloads//Iris.csv',index_col=0)
data.head()


# In[6]:


data.info()


# We can see there is no null values.

# In[7]:


data.describe()


# # Input data visualization

# In[8]:


sns.pairplot(data, hue='Species')


# 
# 
# We can observe that speciesv "Iris Setosa" makes a distinctive cluster in every parameter, while other two species overlap a bit each other.
# 
# # Finding the correlation matrix¶

# In[9]:


data.corr()


# In next step, using heatmap to visulaize data

# In[10]:


sns.heatmap(data.corr())


# We observed that: (i)Petal length is highly related to petal width (ii)Sepal length is not related to sepal width
# 
# # Data preprocessing

# In[11]:


target=data['Species']
df=data.copy()
df=df.drop('Species', axis=1)
df.shape


# # defining the attributes and labels
# 

# In[12]:



X=data.iloc[:, [0,1,2,3]].values
le=LabelEncoder()
data['Species']=le.fit_transform(data['Species'])
y=data['Species'].values
data.shape


# # Trainig the model
# We will now split the data into test and train.

# In[13]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("Traingin split:",X_train.shape)
print("Testin spllit:",X_test.shape)


# Defining Decision Tree Algorithm

# In[14]:


dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
print("Decision Tree Classifier created!")


# # Classification Report and Confusion Matrix

# In[15]:


y_pred=dtree.predict(X_test)
print("Classification report:\n",classification_report(y_test,y_pred))


# The accuracy is 1 or 100% since i took all the 4 features of the iris dataset.

# In[16]:


print("Accuracy:",sm.accuracy_score(y_test,y_pred))


# # Visualization of trained model

# In[17]:


#visualizing the graph
mt.figure(figsize=(20,10))
tree=plot_tree(dtree,feature_names=df.columns,precision=2,rounded=True,filled=True,class_names=target.values)


# The Descision Tree Classifier is created and is visaulized graphically. Also the prediction was calculated using decision tree algorithm and accuracy of the model was evaluated.
# 
# # --------------------------------- End of Code ---------------------------------¶

# In[ ]:




