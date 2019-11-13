#!/usr/bin/env python
# coding: utf-8

# In[41]:


import os
from functools import partial
import numpy as np
import pandas as pd
import seaborn as sns
import math
from scipy import stats

import matplotlib.pyplot as plt



price = pd.read_csv("prices-split-adjusted.csv")

fundamental = pd.read_csv("fundamentals.csv")

fundamental.rename(columns={'Ticker Symbol': 'symbol', 'Period Ending': 'date'}, inplace=True)

fundamental['date']=pd.to_datetime(fundamental['date'])
price['date']=pd.to_datetime(price['date'])

price = price.sort_values([ 'date','symbol']).reset_index()
fundamental = fundamental.sort_values(['date','symbol']).reset_index(drop=True)


#data = pd.merge_asof(price,fundamental,on='date',by='symbol')#,tolerance=pd.Timedelta('90days'))
data = pd.merge(price,fundamental,on=['date','symbol']).reset_index(drop=True)
data['e2p'] = data['Earnings Per Share']/data['close']

data['return'] = (data['close']/data['open']-1)
norm_by_asset_factor = ['Capital Expenditures','Goodwill','Fixed Assets']
data[norm_by_asset_factor] = data[norm_by_asset_factor].div(data['Total Assets'], axis=0)
data['FCF Margin']= data['Net Cash Flow-Operating']/data['Total Revenue']

data.dropna(subset=['return'],inplace=True)#, 'born'])


features = ['e2p', 'Cash Ratio','Gross Margin','Profit Margin','Current Ratio','Operating Margin','Short-Term Debt / Current Portion of Long-Term Debt','FCF Margin']+norm_by_asset_factor



X_train = data[features].values
y_train = data['return'].values



stats.zscore(X_train, axis=1, ddof=1)

X_data = torch.tensor(X_train, dtype=torch.float)
y_data = torch.tensor(y_train, dtype=torch.float)




# In[47]:


from scipy import stats
stats.zscore(X_train, axis=1, ddof=1)


# In[51]:





# In[45]:


len(price.date.unique())


# In[19]:


len(data)


# In[43]:


X_data.size()


# In[26]:


data[norm_by_asset_factor]


# In[29]:


data['Gross Margin']


# In[31]:


data['Gross Profit']/data['Total Revenue']


# In[ ]:




