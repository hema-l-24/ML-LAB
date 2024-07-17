#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score


# In[2]:


iris = datasets.load_iris()
iris.target_names
X = iris.data
y = iris.target
print("Features:\n", iris.feature_names)
print("Classes:\n", iris.target_names)


# In[3]:


df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],columns=iris['feature_names'] + ['target'])
print("\nFirst 5 rows of the dataset:\n", df.head())


# In[4]:


plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Iris Data Sepal Length vs Sepal Width')
plt.show()


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)


# In[6]:


svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(X_train, y_train)


# In[7]:


y_pred = svm_classifier.predict(X_test)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:\n", accuracy_score(y_test, y_pred))


# In[8]:


df.info()


# In[9]:


df['target'].value_counts()


# In[10]:


svm_classifier_rbf = SVC(random_state = 42)
svm_classifier_rbf.fit(X_train, y_train)


# In[11]:


y_pred = svm_classifier_rbf.predict(X_test)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:\n", accuracy_score(y_test, y_pred))


# In[12]:


svm_classifier_poly = SVC(kernel = 'poly', random_state = 42)
svm_classifier_poly.fit(X_train, y_train)


# In[13]:


y_pred = svm_classifier_poly.predict(X_test)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:\n", accuracy_score(y_test, y_pred))

