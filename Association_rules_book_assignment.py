#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules


# In[2]:


books = pd.read_csv("book.csv")
books


# In[5]:


### Appling apriori 
frequent_items = apriori(books, min_support = 0.005,max_len = 3, use_colnames = True)


# In[8]:


## Most frequent items set based on support  
frequent_items.sort_values('support',ascending= False, inplace = True)
frequent_items


# In[9]:


##Building the rules
rules = association_rules(frequent_items,metric="lift",min_threshold = 1)


# In[11]:


##The above code gives us the rules with threshold greater than 1
rules.sort_values('lift', ascending = False, inplace = True)
rules


# In[13]:


####Eliminating the reducdencies in the rules##
def to_list(i):
    return sorted(i)
ma_x = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)
ma_x = ma_x.apply(sorted)
ma_x


# In[19]:


return_rules = list(ma_x)
unique_rules = [list(m) for m in set(tuple(i) for i in return_rules)]
unique_rules


# In[20]:


index_rules = []
for i in unique_rules:
    index_rules.append(return_rules.index(i))


# In[22]:


##Getting the rules without any reducdancies
rules_without_reducdancies = rules.iloc[index_rules,:]
rules_without_reducdancies


# In[23]:


##Sorting them with respect to lift 
rules_without_reducdancies.sort_values('lift', ascending = False, inplace = True)


# In[26]:


### to see only top 10
rules_without_reducdancies.head(10)


# In[27]:


## 3D plot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

support = rules_without_reducdancies["support"]
confidence = rules_without_reducdancies["confidence"]
lift = rules_without_reducdancies["lift"]


# In[28]:


fig = plt.figure()
ax= fig.add_subplot(111, projection = '3d')
ax.scatter(support,confidence,lift)
ax.set_xlabel("Support")
ax.set_ylabel("Confidence")
ax.set_zlabel("lift")


# In[29]:


## Scatter plot
import scipy as sp
plt.scatter(x=support, y=confidence, c=lift , cmap = 'gray')
plt.colorbar()
plt.xlabel("support")
plt.ylabel("confidence")


# ### Total number of rules are 212, wuth minimum support = 0.005 and maximum length = 4 , which are without any reducdancies

# ## Changing the support value to 0.1

# In[75]:


frequent_items1 = apriori(books,min_support = 0.1,max_len = 4 , use_colnames = True)


# In[76]:


#Most frequent items based on the support, decending order
frequent_items1.sort_values('support', ascending = False, inplace = True)
frequent_items1


# In[77]:


##Building rules
rules2 = association_rules(frequent_items1 , metric = 'lift' , min_threshold = 1)


# In[78]:


##Rules2 are the rules which are generated with the minimum threshold as 1
rules2.sort_values('lift',ascending = False , inplace = True)


# In[79]:


###Elimiinating the reducdancies
def to_list1(i):
    return(sorted(i))

ma_x1 = rules2.antecedents.apply(to_list1)+ rules2.consequents.apply(to_list1)
ma_x1 = ma_x1.apply(sorted)
ma_x1


# In[80]:


return_rules1 = list(ma_x1)
unique_rules1 = [list(m) for m in set(tuple(i) for i  in return_rules1 )]


# In[81]:


index_rules1 = []
for i in unique_rules1:
    index_rules1.append(return_rules1.index(i))


# In[82]:


### eliminate rules with reducdancies 
rules_without_reduc = rules2.iloc[index_rules1,:]


# In[83]:


##Sorting the rules
rules_without_reduc.sort_values('lift', ascending = False, inplace = True)


# In[84]:


rules_without_reduc


# In[89]:


rules_without_reduc.shape


# ### A Total of 30 rules

# In[85]:


## 3D plots
support2 = rules_without_reduc["support"]
confidence2 =  rules_without_reduc["confidence"]
lift2 = rules_without_reduc["lift"]


# In[86]:


fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection = '3d')
ax1.scatter(support2,confidence2,lift2)
ax1.set_xlabel("support")
ax1.set_ylabel("confidence")
ax1.set_zlabel("lift")


# In[87]:


## Scatter plot
plt.scatter(support2,confidence2, c =lift2, cmap = 'gray')
plt.colorbar()
plt.xlabel("support");plt.ylabel("confidence")


# In[ ]:





# In[ ]:




