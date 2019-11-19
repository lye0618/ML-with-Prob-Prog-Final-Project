#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 19:17:01 2019

@author: linye
"""

import os
from functools import partial
import numpy as np
import pandas as pd
import seaborn as sns
import math
from scipy import stats
import torch
import matplotlib.pyplot as plt

from torch import nn
import torch.nn.functional as F

import BNN_helper_upd as BNN

np.random.seed(0)



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

X = torch.tensor(X_train, dtype=torch.float)
Y = torch.tensor(y_train, dtype=torch.float)

n_inputs = 11
n_hiddens =[10]
activ_func = F.relu

reg_net = BNN.BNN_REG(n_inputs, n_hiddens, activ_func)

reg_net.Inference(X,Y, n_epochs=1000)

reg_net.get_para()

yhat, yhat_mean, yhat_std = reg_net.predict(X, n_samples=100)