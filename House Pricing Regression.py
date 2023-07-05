#!/usr/bin/env python
# coding: utf-8

# In[495]:


import pandas as pd


# In[496]:


data = pd.read_csv("Real estate.csv")


# In[497]:


data.head()


# In[498]:


data.drop("X1 transaction date",inplace = True,axis =1)


# In[499]:


data.info()


# In[500]:


data.describe()


# In[501]:


data.corr()


# In[502]:


data.head()


# In[503]:


x = data.iloc[:,1:6]
x.head()


# In[504]:


y = data.iloc[:,6]
y.head()


# In[505]:


data.info()


# In[506]:


label=data['Y house price of unit area']


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
fig, ax=plt.subplots(2,1,figsize=(9,12))

# Plot histogram
ax[0].hist(label,bins=100)
ax[0].set_ylabel('Frquency')

ax[0].axvline(label.mean(), color='magenta', linestyle='dashed', linewidth=2)
ax[0].axvline(label.median(), color='cyan', linestyle='dashed', linewidth=2)

# Plot the boxplot
ax[1].boxplot(label, vert=False)
ax[1].set_xlabel('Label')

# Add a title to the Figure
fig.suptitle('Label Distribution')

# Show the figure
fig.show()


# In[507]:


data[data["Y house price of unit area"]>75]


# In[508]:


data = data.drop(data[data["No"].isin([221,271,313])].index)


# In[509]:


data.info()


# In[510]:


import seaborn as sns
corr = data.corr()
sns.heatmap(corr, cmap = 'Blues', annot= True, linewidths=.5);


# In[511]:


from sklearn.model_selection import train_test_split


# In[512]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 1)


# In[513]:


from sklearn.linear_model import LinearRegression


# In[514]:


lm = LinearRegression()


# In[515]:


lm.fit(x_train,y_train)


# In[516]:


lm.score(x_test,y_test)


# In[575]:


y_predlm = lm.predict(x_test)
mselm = mean_squared_error(y_test,y_predlm)
print(mselm)


# In[517]:


lm.coef_


# In[518]:


pd.concat([pd.Series(x.columns),pd.Series(lm.coef_)],axis=1)


# In[519]:


from sklearn.ensemble import GradientBoostingRegressor


# In[565]:


gbr = GradientBoostingRegressor(n_estimators = 48, learning_rate =0.1,max_depth =1,random_state =1)


# In[566]:


gbr.fit(x_train,y_train)


# In[567]:


gbr.score(x_test,y_test)


# In[571]:


y_pred = gbr.predict(x_test)
print(y_pred)


# In[573]:


from sklearn.metrics import mean_squared_error


# In[574]:


mse = mean_squared_error(y_test,y_pred)
print(mse)


# In[ ]:




