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

data.dropna(subset=['return','symbol'],inplace=True)#, 'born'])

data=data.fillna(0)

features = ['e2p', 'Cash Ratio','Gross Margin','Profit Margin','Current Ratio','Operating Margin','Short-Term Debt / Current Portion of Long-Term Debt','FCF Margin']+norm_by_asset_factor



X_train = data[features].values
y_train = data['return'].values



X_train=stats.zscore(X_train, axis=1, ddof=1)
#y_train=stats.zscore(y_train, axis=1, ddof=1)


X_data = torch.tensor(X_train, dtype=torch.float)
y_data = torch.tensor(y_train, dtype=torch.float)


