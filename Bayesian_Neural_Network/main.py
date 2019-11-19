#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:15:58 2019

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
import data as data_process

#model_data, backtest_data = data_process.preprocess()

model_data = pd.read_csv('model_data.csv')

X_train = model_data[model_data.columns[2:-1]].values
y_train = model_data['Rtn1q'].values

#X_train=stats.zscore(X_train, axis=1, ddof=1)
#y_train=stats.zscore(y_train, axis=1, ddof=1)

X = torch.tensor(X_train, dtype=torch.float)
Y = torch.tensor(y_train, dtype=torch.float)

n_inputs = X_train.shape[1]
n_hiddens =[10]
activ_func = F.relu

reg_net = BNN.BNN_REG(n_inputs, n_hiddens, activ_func)

reg_net.Inference(X, Y, lr=0.001, n_epochs=200, batch_size=128)

reg_net.get_para()

yhat, yhat_mean, yhat_std = reg_net.predict(X, n_samples=100)