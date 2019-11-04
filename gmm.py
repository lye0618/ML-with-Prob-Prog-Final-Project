from collections import defaultdict
import numpy as np
import scipy.stats
import torch
from torch.distributions import constraints
from matplotlib import pyplot

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete

import pandas as pd

data = pd.read_csv('/mnt/d/mlpp/data/prices-split-adjusted.csv')
data['return'] = (data['close'] / data['open']) - 1
arr = data['return'].values

y = torch.tensor(data['return'].values)
y = torch.from_numpy(arr)
y = y.type(torch.FloatTensor)


# Trying with fixed number of components
n_comp = 10


# Model Reference: https://pyro.ai/examples/gmm.html
@config_enumerate
def model(data):
    # Global variables.
    weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(n_comp)))
    scale = pyro.sample('scale', dist.LogNormal(0., 2.))
    with pyro.plate('components', n_comp):
        locs = pyro.sample('locs', dist.Normal(0.0004, 0.0002))

    with pyro.plate('data', len(data)):
        # Local variables.
        assignment = pyro.sample('assignment', dist.Categorical(weights))
        pyro.sample('obs', dist.Normal(locs[assignment], scale), obs=data)


# Using AutoDelta guide (MAP estimator)
global_guide = AutoDelta(poutine.block(model, expose=['weights', 'locs', 'scale']))


optim = pyro.optim.Adam({'lr': 0.1, 'betas': [0.8, 0.99]})
elbo = TraceEnum_ELBO(max_plate_nesting=1)
svi = SVI(model, global_guide, optim, loss=elbo)


n_steps = 50
# do gradient steps
for step in range(n_steps):
    svi.step(y)


# Initializations
def initialize(seed):
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    # Initialize weights to uniform.
    pyro.param('auto_weights', 0.5 * torch.ones(n_comp), constraint=constraints.simplex)
    # Assume half of the data variance is due to intra-component noise.
    pyro.param('auto_scale', (y.var() / 2).sqrt(), constraint=constraints.positive)
    # Initialize means from a subsample of data.
    pyro.param('auto_locs', y[torch.multinomial(torch.ones(len(y)) / len(y), n_comp)]);
    loss = svi.loss(model, global_guide, y)
    return loss


# Choose the best among 100 random initializations.
loss, seed = min((initialize(seed), seed) for seed in range(100))
initialize(seed)
print('seed = {}, initial_loss = {}'.format(seed, loss))


gradient_norms = defaultdict(list)
for name, value in pyro.get_param_store().named_parameters():
    value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))


losses = []
for i in range(200 if not smoke_test else 2):
    loss = svi.step(data)
    losses.append(loss)
    print('.' if i % 100 else '\n', end='')


pyplot.figure(figsize=(10,3), dpi=100).set_facecolor('white')
pyplot.plot(losses)
pyplot.xlabel('iters')
pyplot.ylabel('loss')
pyplot.yscale('log')
pyplot.title('Convergence of SVI');

