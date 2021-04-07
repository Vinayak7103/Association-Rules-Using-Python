#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


# In[7]:


movies = pd.read_csv("C:/Users/vinay/Downloads/my_movies.csv")
movies


# In[14]:


movies = movies.iloc[:,[0,1,2,3,4]]


# In[ ]:


#### Imputation to convert the nan values to 0's##
movies.iloc[:,2:5] = movies.iloc[:,2:5].apply(lambda x:x.fillna(0))


# In[12]:


X = pd.get_dummies(movies[['V1','V2','V3','V4','V5']])
X


# In[16]:


######Creating model with dummies of NAN values #####
x_dummies = X.iloc[:,[9,14,16]]


# In[69]:


### Running Apriori algorithm
frequent_items = apriori(x_dummies,min_support = 0.05, max_len =2 , use_colnames = True )
frequent_items.sort_values('support', ascending = False, inplace = True)


# In[70]:


# Building rules
rules_dummies = association_rules(frequent_items, metric = 'lift', min_threshold = 1)
rules_dummies.sort_values('lift',ascending =False,inplace =True)


# In[71]:


### To eliminate redudancy in rules#
def to_list(i):
    return(sorted(i))


# In[72]:


rules_add = rules_dummies.antecedents.apply(to_list) + rules_dummies.consequents.apply(to_list)

rules_add = rules_add.apply(sorted)

rules_set = list(rules_add)


# In[73]:


unique_rules = [list(m) for m in set(tuple(i) for i in rules_set)]
index_rules = []
for i in unique_rules:
    index_rules.append(rules_set.index(i))


# In[74]:


### rules without redudancy##
rules_without_redud = rules_dummies.iloc[index_rules,:]
rules_without_redud 


# In[75]:


### Support and confidence
Support = rules_without_redud['support']
confidence = rules_without_redud['confidence']


# In[76]:


Support


# In[77]:


confidence


# In[78]:


import matplotlib.pyplot as plt

plt.scatter(Support,confidence)
plt.xlabel("Support")
plt.ylabel("Confidence")


# In[79]:


######## Model with other than Zero values#####
x_without_dum = X.iloc[:,[0,1,2,3,4,5,6,7,8,10,11,12,13,15,17]]


# In[80]:


#### Applying Apriori 
frequents_item1 = apriori(x_without_dum, min_support=0.005, max_len = 2, use_colnames=True)
frequents_item1.sort_values('support',ascending = False, inplace = True)


# In[81]:


###Building rules
rules_without = association_rules(frequents_item1, metric='lift', min_threshold =1)
rules_without.sort_values('lift',ascending = False, inplace =True)


# In[82]:


###Eliminate the reducdancy####
def to_list_out(i):
    return(sorted(i))


# In[83]:


rules_out_add = rules_without.antecedents.apply(to_list_out)+rules_without.consequents.apply(to_list_out)

rules_out_add = rules_out_add.apply(sorted)
rules_set_out = list(rules_out_add)


# In[84]:


## rules without redundancy
rules_without_out = rules_without.iloc[index_rules_out,:]


# In[85]:


Support_out = rules_without_out["support"]
Confidence_out = rules_without_out["confidence"]
lift = rules_without_out["lift"]


# In[86]:


Support_out


# In[87]:


#### Plotting 3D plot for support, confidence and lift
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


# In[88]:


fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")
ax.scatter(Support_out, Confidence_out, lift)
ax.set_xlabel("Support_out")
ax.set_ylabel("Confidence_out")
ax.set_zlabel("lift")


# In[89]:


#### scatter plot for rules for support, confidence and lift
import matplotlib.pyplot as plt
import scipy as sp

plt.scatter(Support_out, Confidence_out,c= lift,cmap='gray')
plt.colorbar()
plt.xlabel("Support")
plt.ylabel("Confidence")


# In[ ]:





# In[ ]:




