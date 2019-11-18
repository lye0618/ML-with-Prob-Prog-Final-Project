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
from pyro.infer.autoguide import AutoDelta, AutoDiagonalNormal, AutoMultivariateNormal
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
from pyro.infer.autoguide.initialization import init_to_mean
from loader import read_data
from preprocessing import X_data


class GMM(object):

    # Set device to CPU
    device = torch.device('cpu')

    def __init__(self, n_comp=10, infr='svi', subsample=False):
        assert infr == 'svi' or infr == 'mcmc', 'Only svi and mcmc supported'
        # Load data
        # df = read_data(data_type='nyse')
        data = X_data
        self.tensor = data.type(torch.FloatTensor)
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
            if subsample:
                self.mcmc_subsample = 0.1
                self.n_obs = int(self.shape[0] * self.mcmc_subsample)
            else:
                self.n_obs = self.shape[0]
            # Need to subsample in numpy array because
            # sampling using multinomial takes ages
            # self.tensor = torch.from_numpy(np.random.choice(data, self.n_obs)).type(torch.FloatTensor)

        # Initialize model
        self.model()

    ##################
    # Model definition
    ##################
    @config_enumerate
    def model_working(self):
        # Global variables
        weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(self.n_comp)))
        scale = pyro.sample('scale', dist.LogNormal(0., 2.))

        with pyro.plate('components', self.n_comp):
            locs = pyro.sample('locs', dist.MultivariateNormal(torch.rand(self.shape[1]), torch.eye(self.shape[1])))

        with pyro.plate('data', self.shape[0]):
            # Local variables.
            assignment = pyro.sample('assignment', dist.Categorical(weights))
            pyro.sample('obs', dist.MultivariateNormal(locs[assignment], torch.eye(self.shape[1])*scale),
                        obs=self.tensor)

    @config_enumerate
    def model(self):
        # Global variables
        weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(self.n_comp)))
        scale_temp = pyro.sample('scale', dist.LogNormal(0., 2.))
        scale = scale_temp * torch.eye(self.shape[1])
        # scale = pyro.sample('scale', dist.LogNormal(0., 2.))

        with pyro.plate('components', self.n_comp):
            locs = pyro.sample('locs', dist.MultivariateNormal(torch.rand(self.shape[1]), torch.eye(self.shape[1])))

        with pyro.plate('data', self.shape[0]):
            # Local variables.
            assignment = pyro.sample('assignment', dist.Categorical(weights))
            y = pyro.sample('obs', dist.MultivariateNormal(locs[assignment], scale),
                        obs=self.tensor)
            return y


    ##################
    # SVI
    ##################
    def guide_autodelta(self):
        self.guide = AutoDelta(poutine.block(self.model, expose=['weights', 'locs', 'scale']))
                               # init_loc_fn=init_to_mean)

    def guide_autodiagnorm(self):
        self.guide = AutoDiagonalNormal(poutine.block(self.model, expose=['weights', 'locs', 'scale']))
                                        # init_loc_fn=init_to_mean)

    def guide_multivariatenormal(self):
        self.guide = AutoMultivariateNormal(poutine.block(self.model, expose=['weights', 'locs', 'scale']))
                                            # init_loc_fn=init_to_mean)

    def guide_manual(self):
        # Define priors
        weights_alpha = pyro.param('weights_alpha', torch.randn(self.n_comp), constraint=constraints.positive)
        scale_loc = pyro.param('scale_loc', torch.rand(1), constraint=constraints.positive)
        scale_scale = pyro.param('scale_scale', torch.rand(1), constraint=constraints.positive)
        loc_loc = pyro.param('scale_loc', torch.rand(self.shape[1]), constraint=constraints.positive)
        loc_scale = pyro.param('scale_scale', torch.rand(1), constraint=constraints.positive)

        # Global variables
        weights = pyro.sample('weights', dist.Dirichlet(weights_alpha))
        scale = pyro.sample('scale', dist.LogNormal(scale_loc, scale_scale))
        with pyro.plate('components', self.n_comp):
            self.locs = pyro.sample('locs', dist.MultivariateNormal(loc_loc, torch.eye(self.shape[1]) * loc_scale))

        with pyro.plate('data', self.shape[0]):
            # Local variables.
            assignment = pyro.sample('assignment', dist.Categorical(self.weights))

    def optimizer(self):
        self.optim = pyro.optim.Adam({'lr': 0.1, 'betas': [0.8, 0.99]})

    def initialize(self, seed):
        self.set_seed(seed)
        self.clear_params()

        # Initialize weights to uniform.
        pyro.param('auto_weights', (1./self.n_comp) * torch.ones(self.n_comp), constraint=constraints.simplex)

        # Assume half of the data variance is due to intra-component noise.
        pyro.param('auto_scale', (self.tensor.var() / 2).sqrt(), constraint=constraints.positive)
        # pyro.param('auto_scale', torch.eye(self.shape[1]) * torch.rand(self.shape[1]), constraint=constraints.positive)

        # Initialize means from a subsample of data.
        pyro.param('auto_locs', self.tensor[torch.multinomial(torch.ones(self.shape[0]) / self.shape[0], self.n_comp)])
        return self.run_svi()

    def init_svi(self):
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo_loss)

    def run_svi(self):
        if self.guide is None:
            self.guide = self.guide_manual
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
                print('.' if i % 100 else '\n', end='')
                end = time.time()
                self.svi_time = (end - start)
            return losses
        else:
            start = time.time()
            self.run_mcmc()
            end = time.time()
            self.mcmc_time = (end - start)
            return self.get_mcmc_samples()

    # Get posterior responsibilities
    def get_posterior_resp(self):
        '''
        Formula:
        k: cluster index
        p(c=k|x) = w_k * N(x|mu_k, sigma_k) / sum(w_k * N(x|mu_k, sigma_k))
        '''
        prob_list = []
        distri = dist.MultivariateNormal(self.locs, torch.eye(self.shape[1]) * self.scale)
        for d in self.tensor:
            numerator = self.weights * torch.exp(distri.log_prob(d))
            denom = numerator.sum()
            probs = numerator / denom
            prob_list.append(probs)

        resp = torch.stack(prob_list)

        # TODO: Debug why nan
        resp[torch.isnan(resp)] = 0
        return resp

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