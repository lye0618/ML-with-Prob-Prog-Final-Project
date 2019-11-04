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



price = pd.read_csv("prices-split-adjusted.csv")

fundamental = pd.read_csv("fundamentals.csv")

fundamental.rename(columns={'Ticker Symbol': 'symbol', 'Period Ending': 'date'}, inplace=True)

fundamental['date']=pd.to_datetime(fundamental['date'])
price['date']=pd.to_datetime(price['date'])

price = price.sort_values(['symbol', 'date']).reset_index()
fundamental = fundamental.sort_values(['symbol', 'date']).reset_index()


#data = pd.merge_asof(price,fundamental,on='date',by='symbol',tolerance=pd.Timedelta('90days'))
data = pd.merge(price,fundamental,on=['date','symbol'])


data['return'] = (data['close']/data['open']-1)
x_train = data[['open','close','high','low']].values
y_train = data['return'].values

x_data = torch.tensor(x_train, dtype=torch.float)
x_data = torch.tensor(y_train, dtype=torch.float)



