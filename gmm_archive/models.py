# Imports
from collections import defaultdict
import time
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoDelta, AutoDiagonalNormal
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
from loader import read_data


class GMM(object):

    # Set device to CPU
    device = torch.device('cpu')

    def __init__(self, n_comp=10, infr='svi'):
        assert infr == 'svi' or infr == 'mcmc', 'Only svi or mcmc supported'
        # Load data
        df = read_data(data_type='nyse')
        data = df['return'].values
        self.tensor = torch.from_numpy(data).type(torch.FloatTensor)
        self.n_comp = n_comp
        self.infr = infr
        self.shape = self.tensor.shape
        self.params = None
        self.weights = None
        self.locs = None
        self.scale = None
        self.mcmc_time = None
        self.svi_time = None
        print(f'Initializing object for inference method {self.infr}')
        if self.infr == 'svi':
            self.guide = None
            self.optim = Adam({'lr': 0.1, 'betas': [0.8, 0.99]})
            self.svi = None
            self.svi_itr = 100
            self.elbo_loss = TraceEnum_ELBO(max_plate_nesting=1)
        else:
            self.num_samples = 250
            self.mcmc = None
            self.warmup_steps = 50
            self.mcmc_subsample = 0.1
            self.n_obs = int(self.shape[0] * self.mcmc_subsample)
            # Need to subsample in numpy array because
            # sampling using multinomial takes ages
            self.tensor = torch.from_numpy(np.random.choice(data, self.n_obs)).type(torch.FloatTensor)

        # Initialize model
        self.model()

    ##################
    # Model definition
    ##################
    @config_enumerate
    def model(self):
        # Global variables.
        self.weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(self.n_comp)))
        self.scale = pyro.sample('scale', dist.LogNormal(0., 2.))
        with pyro.plate('components', self.n_comp):
            self.locs = pyro.sample('locs', dist.Normal(0.0004, 0.0002))

        with pyro.plate('data', len(self.tensor)):
            # Local variables.
            assignment = pyro.sample('assignment', dist.Categorical(self.weights))
            y = pyro.sample('obs', dist.Normal(self.locs[assignment], self.scale), obs=self.tensor)

        return y

    ##################
    # SVI
    ##################
    def guide_autodelta(self):
        self.guide = AutoDelta(poutine.block(self.model, expose=['weights', 'locs', 'scale']))

    def guide_autodiagnorm(self):
        self.guide = AutoDiagonalNormal(poutine.block(self.model, expose=['weights', 'locs', 'scale']))

    def guide_manual(self):
        # self.guide = AutoDelta(poutine.block(self.model, expose=['weights', 'locs', 'scale']))
        pass

    def optimizer(self):
        self.optim = pyro.optim.Adam({'lr': 0.1, 'betas': [0.8, 0.99]})

    def initialize(self, seed):
        self.set_seed(seed)
        self.clear_params()

        # Initialize weights to uniform.
        pyro.param('auto_weights', 0.5 * torch.ones(self.n_comp), constraint=constraints.simplex)

        # Assume half of the data variance is due to intra-component noise.
        pyro.param('auto_scale', (self.tensor.var() / 2).sqrt(), constraint=constraints.positive)

        # Initialize means from a subsample of data.
        pyro.param('auto_locs', self.tensor[torch.multinomial(torch.ones(self.shape[0]) / self.shape[0], self.n_comp)])
        return self.run_svi()

    def init_svi(self):
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo_loss)

    def run_svi(self):
        self.init_svi()
        loss = self.svi.loss(self.model, self.guide)
        return loss

    def best_start(self):
        # Choose the best among 100 random initializations.
        print("Determining best seed for initialization")
        loss, seed = min((self.initialize(seed), seed) for seed in range(100))
        self.initialize(seed)
        print("Best seed determined after 100 random initializations:")
        print('seed = {}, initial_loss = {}'.format(seed, loss))

    def params(self):
        self.params = pyro.get_param_store()
        return self.params

    def register_params(self):
        gradient_norms = defaultdict(list)
        for name, value in self.params.named_parameters():
            value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))

    def get_svi_estimates(self):
        estimates = self.guide(self.tensor)
        self.weights = estimates['weights']
        self.locs = estimates['locs']
        self.scale = estimates['scale']
        return self.weights, self.locs, self.scale

    # TODO This is BS, make it vectorized
    def get_posterior_resp(self):
        '''
        Formula:
        k: cluster index
        p(c=k|x) = w_k * N(x|mu_k, sigma_k) / sum(w_k * N(x|mu_k, sigma_k))
        '''
        prob_list = []
        distri = dist.Normal(self.locs, self.scale)
        for d in self.tensor:
            numerator = self.weights * distri.log_prob(d)
            denom = numerator.sum()
            probs = numerator / denom
            prob_list.append(probs)

        final = torch.stack(prob_list)
        return final

    ##################
    # MCMC
    ##################
    def init_mcmc(self, seed=42):
        self.set_seed(seed)
        kernel = NUTS(self.model)
        self.mcmc = MCMC(kernel, num_samples=self.num_samples, warmup_steps=self.warmup_steps)
        print("Initialized MCMC with NUTS kernal")

    def run_mcmc(self):
        self.clear_params()
        print("Initializing MCMC")
        self.init_mcmc()
        print(f'Running MCMC using NUTS with num_obs = {self.n_obs}')
        self.mcmc.run()

    def get_mcmc_samples(self):
        return self.mcmc.get_samples()

    ##################
    # Inference
    ##################
    def inference(self):
        if self.infr == 'svi':
            start = time.time()
            # Initialize with best seed
            self.best_start()
            # Run SVI iterations
            print("Running SVI iterations")
            losses = []
            for i in range(self.svi_itr):
                loss = self.svi.step()
                losses.append(loss)
                print('.' if i % (int(self.svi_itr/2)) else '\n', end='')
                end = time.time()
                self.svi_time = (end - start)
            return losses
        else:
            start = time.time()
            self.run_mcmc()
            end = time.time()
            self.mcmc_time = (end - start)
            return self.get_mcmc_samples()

    ##################
    # Generate stats
    ##################
    def generate_stats(self):
        if self.svi is not None:
            svi_stats = dict({'num_samples': self.shape[0],
                              'num_iterations': self.svi_itr,
                              'exec_time': self.svi_time})
        else:
            svi_stats = None

        if self.mcmc is not None:
            mcmc_stats = dict({'num_samples': self.shape[0]*self.mcmc_subsample,
                               'exec_time': self.mcmc_time,
                               'num_samples_generated': self.num_samples,
                               'warmup_steps': self.warmup_steps})
        else:
            mcmc_stats = None

        return [svi_stats, mcmc_stats]

    ##################
    # Static Methods
    #################
    @staticmethod
    def set_seed(seed):
        pyro.set_rng_seed(seed)

    @staticmethod
    def clear_params():
        pyro.clear_param_store()

    @staticmethod
    def plot_svi_convergence(losses):
        plt.figure(figsize=(10, 3), dpi=100).set_facecolor('white')
        plt.plot(losses)
        plt.xlabel('iters')
        plt.ylabel('loss')
        plt.title('Convergence of SVI')
        plt.plot()