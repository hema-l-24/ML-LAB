#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
import sklearn.metrics as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


dataset=load_iris()


# In[4]:


X=pd.DataFrame(dataset.data)
X.columns=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']


# In[5]:


y=pd.DataFrame(dataset.target)
y.columns=['Targets']


# In[6]:


plt.figure(figsize=(14,7))
colormap=np.array(['red','lime','black'])


# In[7]:


plt.subplot(1,3,1)
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[y.Targets],s=40)
plt.title('Real')


# In[8]:


plt.subplot(1,3,2)
model=KMeans(n_clusters=3)
model.fit(X)
predY=np.choose(model.labels_,[0,1,2]).astype(np.int64)
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[predY],s=40)
plt.title('KMeans')


# In[9]:


scaler=preprocessing.StandardScaler()
scaler.fit(X)
xsa=scaler.transform(X)
xs=pd.DataFrame(xsa,columns=X.columns)
gmm=GaussianMixture(n_components=3)
gmm.fit(xs)
y_cluster_gmm=gmm.predict(xs)
plt.subplot(1,3,3)
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[y_cluster_gmm],s=40)
plt.title('GMM Classification')


# In[10]:


model.labels_


# In[11]:


kmeans_labels=model.labels_


# In[12]:


kmeans_labels


# In[13]:


gmm.fit_predict(X)


# In[14]:


gmm_labels=gmm.fit_predict(X)


# In[15]:


gmm_labels


# In[16]:


from sklearn.metrics import adjusted_rand_score,silhouette_score


# In[17]:


silhouette_kmeans=silhouette_score(y,kmeans_labels)
silhouette_gmm=silhouette_score(y,gmm_labels)


# In[18]:


print(f'Silhouette Score for KMeans : {silhouette_kmeans}')
print(f'Silhouette Score for GMM : {silhouette_gmm}')


# In[ ]:




