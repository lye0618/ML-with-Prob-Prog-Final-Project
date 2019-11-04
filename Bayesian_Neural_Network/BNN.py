#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 21:13:42 2019

@author: linye
"""

###Reference: https://alsibahi.xyz/snippets/2019/06/15/pyro_mnist_bnn_kl.html


import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import BNN_helper as BNN

np.random.seed(0)


"""
#Test classification networks

X = np.random.normal(size=(1000,10))

Y = np.floor(5*np.random.rand(1000,1)).astype('int')

n_inputs = 10
n_hiddens =[10]
n_classes =5
activ_func =F.relu

clf_net = BNN.BNN_CLF(n_inputs, n_hiddens, n_classes, activ_func)

clf_net.Inference(torch.from_numpy(X).float(),torch.from_numpy(Y).float())

yhat, yhat_mean = clf_net.predict(torch.from_numpy(X).float(), n_samples=100)

"""


#Test Regression networks

X = np.random.normal(size=(1000,10))

Y = np.random.normal(size=(1000,1))

n_inputs = 10
n_hiddens =[10]
activ_func = F.relu

reg_net = BNN.BNN_REG(n_inputs, n_hiddens, activ_func)

reg_net.Inference(torch.from_numpy(X).float(),torch.from_numpy(Y).float())

yhat, yhat_mean = reg_net.predict(torch.from_numpy(X).float(), n_samples=100)
