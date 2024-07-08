#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


get_ipython().system('dir dat')


# In[5]:


df=pd.read_csv("ENJOYSPORT.csv")


# In[6]:


df.head()


# In[7]:


df.shape


# In[9]:


attribute=np.array(df)[:,:-1]


# In[10]:


print(attribute)


# In[12]:


target=np.array(df)[:,-1]


# In[13]:


print(target)


# In[15]:


def train(att,tar):
    for i,val in enumerate(tar):
        if val==1:
            specific_h=att[i].copy()
            break
    for i,val in enumerate(att):
        if tar[i]==1:
            for x in range(len(specific_h)):
                if val[x]!=specific_h[x]:
                    specific_h[x]='?'
                else:
                    pass
    return specific_h     
print(train(attribute,target))
                


# In[ ]:




