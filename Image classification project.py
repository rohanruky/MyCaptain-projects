#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


data = pd.read_csv('sign_mnist_test.csv')


# In[10]:


data.head()


# In[37]:


a = data.iloc[5,1:].values


# In[38]:


a = a.reshape(28,28).astype('uint8')
plt.imshow(a)


# In[39]:


df_x = data.iloc[:,1:]
df_y = data.iloc[:,0]


# In[40]:


x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state=4)


# In[41]:


x_train.head()


# In[42]:


y_train.head()


# In[43]:


rf = RandomForestClassifier(n_estimators=100)


# In[44]:


rf.fit(x_train, y_train)


# In[45]:


pred = rf.predict(x_test)


# In[46]:


pred


# In[47]:


s = y_test.values

count = 0 
for i in range(len(pred)):
    if pred[i] == s[i]:
        count = count+1


# In[48]:


count


# In[49]:


len(pred)


# In[50]:


1434/1435


# In[ ]:




