#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd


# In[19]:


get_ipython().system('dir Dat*')


# In[20]:


df=pd.read_csv("Dataset10.csv")


# In[21]:


df.head()


# In[22]:


df.shape


# In[23]:


data=df.drop(['Day'],axis=1)
data


# In[24]:


attribute=np.array(data)[:,:-1]
print(attribute)


# In[25]:


target=np.array(data)[:,-1]
print(target)


# In[26]:


def train(att,tar):
    for i,val in enumerate(tar):
        if val=='Yes':
            specific_h=att[i].copy()
            break
    for i,val in enumerate(att):
        if tar[i]=='Yes':
            for x in  range(len(specific_h)):
                if val[x]!=specific_h[x]:
                    specific_h[x]='?'
                else:
                    pass
    return specific_h


# In[27]:


print(train(attribute,target))


# In[ ]:




