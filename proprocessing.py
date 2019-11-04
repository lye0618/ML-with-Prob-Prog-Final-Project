#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from functools import partial
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import math

import matplotlib.pyplot as plt

import pyro
from pyro.distributions import Normal, Uniform, Delta
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.distributions.util import logsumexp
from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, TracePredictive
from pyro.infer.mcmc import MCMC, NUTS
import pyro.optim as optim
import pyro.poutine as poutine



data = pd.read_csv("prices-split-adjusted.csv")


data.head(10)


data['return'] = (data['close']/data['open']-1)
x_train = data[['open','close','high','low']].values
y_train = data['return'].values

x_data = torch.tensor(x_train, dtype=torch.float)
x_data = torch.tensor(y_train, dtype=torch.float)


# In[ ]:




