#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[2]:


load_iris()


# In[3]:


iris=load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target


# In[4]:


X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)


# In[5]:


k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)


# In[6]:


y_pred = knn.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f'Accuracy: {accuracy:.2f}')


# In[7]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: \n", cm)


# In[8]:


correct_predictions = []
wrong_predictions = []
for i in range(len(y_test)):
    if y_test.iloc[i] == y_pred[i]:
        correct_predictions.append((X_test.iloc[i].tolist(), y_test.iloc[i], y_pred[i]))
    else:
        wrong_predictions.append((X_test.iloc[i].tolist(), y_test.iloc[i], y_pred[i]))
print("\nCorrect Predictions:")
for cp in correct_predictions:
    print(f"Features: {cp[0]}, True Label: {cp[1]}, Predicted Label: {cp[2]}")
print("\nWrong Predictions:")
for wp in wrong_predictions:
    print(f"Features: {wp[0]}, True Label: {wp[1]}, Predicted Label: {wp[2]}")

