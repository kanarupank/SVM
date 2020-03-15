#!/usr/bin/env python
# coding: utf-8

# In[16]:


print('SVM on Wisconsin Brest Cancer Data')


# In[41]:


# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[42]:


#column names
col_names = ['Code Number', 'Clump Thickness','Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

# load dataset
wbcd = pd.read_csv('wbcd.csv', header=None, names=col_names)
wbcdReplacedData = pd.read_csv('wbcdReplacedData.csv', header=None, names=col_names)

#list first 5 rows
wbcd.head()



# In[36]:


wbcdReplacedData.head()


# In[43]:


wbcd.dtypes


# In[43]:


#split dataset in features and target variable
feature_cols = [ 'Clump Thickness','Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
features= wbcd[feature_cols] # Features
result = wbcd.Class # Target variable
featuresReplacedData= wbcdReplacedData[feature_cols] # Features all data
resultReplacedData = wbcdReplacedData.Class # Target variable all data


# In[63]:


# split X and y into training and teting sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, result, test_size = 0.35)
X_train_, X_test_, y_train_, y_test_ = train_test_split(featuresReplacedData, resultReplacedData, test_size = 0.35)

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)


y_pred=svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[69]:


svclassifier.fit(X_train_, y_train_)
y_pred_=svclassifier.predict(X_test_)

print(confusion_matrix(y_test_,y_pred_))
print(classification_report(y_test_,y_pred_))

