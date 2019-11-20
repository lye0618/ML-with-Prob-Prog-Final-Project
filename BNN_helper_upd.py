#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 09:21:49 2019

@author: linye
"""
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import pyro
from pyro.distributions import Normal, Categorical, Uniform
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
# check log_prob (to_event) shape is correct or not
pyro.enable_validation(True)


class NN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[10],
                 output_size=1, activ_func=F.relu):
        super(NN, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.activ_func = activ_func
        setattr(self, 'h0', nn.Linear(input_size, hidden_sizes[0]))
        for i in range(len(hidden_sizes)-1):
            setattr(self,
                    'h'+str(i+1),
                    nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))

        self.out = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        output = self.activ_func(self.h0(x))
        for i in range(len(self.hidden_sizes)-1):
            output = self.activ_func(getattr(self, 'h'+str(i+1))(output))

        output = self.out(output)
        return output


class BNN_CLF(object):

    def __init__(self, n_inputs, n_hiddens=[10],
                 n_classes=10, activ_func=F.relu):
        super(BNN_CLF, self).__init__()
        assert len(n_hiddens) > 0
        self.net = NN(n_inputs, n_hiddens, n_classes, activ_func)
        # self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def model(self, features, target):

        def normal_prior(x):
            return Normal(torch.zeros_like(x), torch.ones_like(x)).to_event(
                x.dim())
        self.priors = {}

        for i in range(len(self.net.hidden_sizes)):
            self.priors['h'+str(i)+'.weight'] = normal_prior(
                getattr(self.net, 'h'+str(i)).weight)
            self.priors['h'+str(i)+'.bias'] = normal_prior(
                getattr(self.net, 'h'+str(i)).bias)

        self.priors['out'+'.weight'] = normal_prior(self.net.out.weight)
        self.priors['out'+'.bias'] = normal_prior(self.net.out.bias)

        # lift module parameters to random variables sampled from the priors
        lifted_module = pyro.random_module("module", self.net, self.priors)
        # sample a regressor (which also samples w and b)
        model_sample = lifted_module()
        # print(model_sample)

        with pyro.plate("data", len(target)):

            # yhat = self.log_softmax(model_sample(features))
            # target is not one-hot encoded
            # pyro.sample("obs",
            #             Categorical(logits=yhat), obs=target)

            yhat = self.softmax(model_sample(features))

            # target is not one-hot encoded
            pyro.sample("obs", Categorical(probs=yhat), obs=target)
            return yhat

    def guide(self, features, target):

        def normal_posterior(x, name):
            hw_mu = pyro.param(name + 'mu', torch.randn_like(x))
            hw_sigma = F.softplus(pyro.param(name + 'sigma',
                                             torch.randn_like(x)))
            return Normal(loc=hw_mu, scale=hw_sigma).to_event(x.dim())

        self.est_priors = {}

        for i in range(len(self.net.hidden_sizes)):
            self.est_priors['h'+str(i)+'.weight'] = normal_posterior(
                getattr(self.net, 'h'+str(i)).weight, 'h'+str(i)+'_w_')
            self.est_priors['h'+str(i)+'.bias'] = normal_posterior(
                getattr(self.net, 'h'+str(i)).bias, 'h'+str(i)+'_b_')

        self.est_priors['out'+'.weight'] = normal_posterior(
            self.net.out.weight, 'out'+'_w_')
        self.est_priors['out'+'.bias'] = normal_posterior(
            self.net.out.bias, 'out'+'_b_')

        # lift module parameters to random variables sampled from the priors
        lifted_module = pyro.random_module("module", self.net, self.est_priors)

        return lifted_module()

    def Inference(self, X_data, Y_data, loss=Trace_ELBO(),
                  optimizer=Adam, lr=0.01, n_epochs=100, batch_size=128):

        self.optim = optimizer({"lr": lr})
        self.svi = SVI(self.model, self.guide, self.optim, loss=loss)

        num_examples = len(X_data)

        for epoch in range(n_epochs):
            total_loss = 0.0
            indices = np.arange(num_examples)
            np.random.shuffle(indices)

            X_data_i = X_data[indices]
            Y_data_i = Y_data[indices]

            for i in range(num_examples // batch_size):
                X_data_minibatch = X_data_i[i*batch_size:(i+1)*batch_size]
                Y_data_minibatch = Y_data_i[i*batch_size:(i+1)*batch_size]

                loss = self.svi.step(X_data_minibatch, Y_data_minibatch)

                total_loss += loss / num_examples

            print(f'Epoch {epoch+1} : Loss {total_loss}')

    def predict(self, X_data, n_samples):

        sampled_models = [self.guide(None, None) for _ in range(n_samples)]
        # shape [num_samples, num_models]
        yhats = torch.stack(
            [self.softmax(model(X_data)).data for model in sampled_models],
            dim=0)
        mean = torch.mean(yhats, dim=0)
        std = torch.std(yhats, dim=0)
        return yhats.numpy(), mean.numpy(), std.numpy()


class BNN_REG(object):

    def __init__(self, n_inputs, n_hiddens=[10], activ_func=F.relu):
        super(BNN_REG, self).__init__()
        assert len(n_hiddens) > 0
        self.net = NN(n_inputs, n_hiddens, 1, activ_func)

    def model(self, features, target):

        def normal_prior(x):
            return Normal(torch.zeros_like(x), torch.ones_like(x)).to_event(
                x.dim())

        self.priors = {}

        for i in range(len(self.net.hidden_sizes)):
            self.priors['h'+str(i)+'.weight'] = normal_prior(
                getattr(self.net, 'h'+str(i)).weight)
            self.priors['h'+str(i)+'.bias'] = normal_prior(
                getattr(self.net, 'h'+str(i)).bias)

        self.priors['out'+'.weight'] = normal_prior(self.net.out.weight)
        self.priors['out'+'.bias'] = normal_prior(self.net.out.bias)

        # lift module parameters to random variables sampled from the priors
        lifted_module = pyro.random_module("module", self.net, self.priors)
        # sample a regressor (which also samples w and b)
        model_sample = lifted_module()

        out_sigma = pyro.sample("sigma", Uniform(0., 10.))

        # precision = pyro.sample("precision", Uniform(0., 10.))
        # out_sigma = 1 / precision

        with pyro.plate("data", len(target)):

            target_mean = model_sample(features).squeeze(-1)
            # target is not one-hot encoded
            pyro.sample("obs", Normal(target_mean, out_sigma), obs=target)

            return target_mean

    def guide(self, features, target):

        def normal_posterior(x, name):
            hw_mu = pyro.param(name + 'mu', torch.randn_like(x))
            hw_sigma = F.softplus(
                pyro.param(name + 'sigma', torch.randn_like(x)))
            return Normal(loc=hw_mu, scale=hw_sigma).to_event(x.dim())

        self.est_priors = {}

        for i in range(len(self.net.hidden_sizes)):
            self.est_priors['h'+str(i)+'.weight'] = normal_posterior(
                getattr(self.net, 'h'+str(i)).weight, 'h'+str(i)+'_w_')
            self.est_priors['h'+str(i)+'.bias'] = normal_posterior(
                getattr(self.net, 'h'+str(i)).bias, 'h'+str(i)+'_b_')

        self.est_priors['out'+'.weight'] = normal_posterior(
            self.net.out.weight, 'out'+'_w_')
        self.est_priors['out'+'.bias'] = normal_posterior(
            self.net.out.bias, 'out'+'_b_')

        # lift module parameters to random variables sampled from the priors
        lifted_module = pyro.random_module("module", self.net, self.est_priors)

        # sigma_loc = pyro.param('sigma_loc', torch.randn(1))
        # out_sigma = pyro.sample("sigma",
        #                         LogNormal(sigma_loc, torch.tensor(0.05)))

        # sigma_loc = F.softplus(pyro.param('sigma_loc', torch.randn(1)))
        # out_sigma = pyro.sample("sigma",
        #                         Normal(sigma_loc, torch.tensor(0.1)))
        # print("out_sigma", out_sigma)

        # alpha = pyro.param(
        #     "alpha", torch.tensor(5.0), constraint = constraints.positive)
        # beta  = pyro.param(
        #     "beta", torch.tensor(0.5), constraint = constraints.positive)

        # precision = pyro.sample("precision", Gamma(alpha, beta))
        return lifted_module()

    def Inference(self, X_data, Y_data, loss=Trace_ELBO(),
                  optimizer=Adam, lr=0.01, n_epochs=100, batch_size=128):

        self.optim = optimizer({"lr": lr})
        self.svi = SVI(self.model, self.guide, self.optim, loss=loss)
        # clear trained parameters
        pyro.clear_param_store()

        num_examples = len(X_data)

        for epoch in range(n_epochs):
            total_loss = 0.0
            indices = np.arange(num_examples)
            np.random.shuffle(indices)

            X_data_i = X_data[indices]
            Y_data_i = Y_data[indices]

            for i in range(num_examples // batch_size):
                X_data_minibatch = X_data_i[i*batch_size:(i+1)*batch_size]
                Y_data_minibatch = Y_data_i[i*batch_size:(i+1)*batch_size]

                loss = self.svi.step(X_data_minibatch, Y_data_minibatch)
                total_loss += loss / num_examples

            # total_loss = self.svi.evaluate_loss(X_data_i, Y_data_i)

            print(f'Epoch {epoch+1} : Loss {total_loss}')

    def get_para(self):
        for name, value in pyro.get_param_store().items():
            print(name, pyro.param(name))

    def predict(self, X_data, n_samples):

        sampled_models = [self.guide(None, None) for _ in range(n_samples)]
        # shape [num_samples, num_models]
        yhats = torch.cat([model(X_data).data for model in sampled_models],
                          dim=1)
        mean = torch.mean(yhats, dim=1)
        return mean.numpy()
        # std = torch.std(yhats, dim=1)
        # return yhats.numpy(), mean.numpy(), std.numpy()
