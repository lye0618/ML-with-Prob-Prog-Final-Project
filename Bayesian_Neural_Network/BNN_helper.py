#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 22:42:24 2019

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

import pyro.poutine as poutine

pyro.enable_validation(True)


class NN(nn.Module):
    
    def __init__(self, input_size, hidden_sizes=[10], output_size=1, activ_func=F.relu):
        
        super(NN, self).__init__()
        
        #self.fc_layers = {}
        
        self.hidden_sizes = hidden_sizes
        self.activ_func = activ_func
        
        #self.fc_layers['h0'] = nn.Linear(input_size, hidden_sizes[0])
        
        #for i in range(len(hidden_sizes)-1):
            
            #self.fc_layers['h'+str(i+1)] = nn.Linear(hidden_sizes[i], hidden_sizes[i+1])
        
        setattr(self, 'h0', nn.Linear(input_size, hidden_sizes[0]))
        
        for i in range(len(hidden_sizes)-1):
            
            setattr(self, 'h'+str(i+1), nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        
        self.out = nn.Linear(hidden_sizes[-1], output_size)
        
        
    def forward(self, x):
        
        output = self.activ_func(self.h0(x))
        
        for i in range(len(self.hidden_sizes)-1):
            output = self.activ_func(getattr(self, 'h'+str(i+1))(output))
        
        output = self.out(output)
        
        return output


class BNN_CLF(object):
    
    def __init__(self, n_inputs, n_hiddens=[10], n_classes=10, activ_func=F.relu):
        
        super(BNN_CLF, self).__init__()
        assert len(n_hiddens) > 0
        
        self.net = NN(n_inputs, n_hiddens, n_classes, activ_func)
        
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def model(self, features, target):
        
        self.priors = {}
        
        for i in range(len(self.net.hidden_sizes)):
            self.priors['h'+str(i)+'.weight'] = Normal(loc=torch.zeros_like(getattr(self.net,'h'+str(i)).weight), scale=torch.ones_like(getattr(self.net,'h'+str(i)).weight)).to_event(2)
            self.priors['h'+str(i)+'.bias'] = Normal(loc=torch.zeros_like(getattr(self.net,'h'+str(i)).bias), scale=torch.ones_like(getattr(self.net,'h'+str(i)).bias)).to_event(1)
        
        self.priors['out'+'.weight'] = Normal(loc=torch.zeros_like(self.net.out.weight), scale=torch.ones_like(self.net.out.weight)).to_event(2)
        self.priors['out'+'.bias'] = Normal(loc=torch.zeros_like(self.net.out.bias), scale=torch.ones_like(self.net.out.bias)).to_event(1)
        
        # lift module parameters to random variables sampled from the priors
        lifted_module = pyro.random_module("module", self.net, self.priors)
        # sample a regressor (which also samples w and b)
        model_sample = lifted_module()
        #print(model_sample)
        
        with pyro.plate("data", len(target)):
            
            yhat = self.log_softmax(model_sample(features))
            pyro.sample("obs", Categorical(logits=yhat), obs=target) #target is not one-hot encoded
    
    def guide(self, features, target):
        
        self.est_priors = {}
        
        for i in range(len(self.net.hidden_sizes)):
            hw_mu = pyro.param('h'+str(i)+'_w_' + 'mu', torch.randn_like(getattr(self.net,'h'+str(i)).weight))
            hw_sigma = F.softplus(pyro.param('h'+str(i)+'_w_' + 'sigma', torch.randn_like(getattr(self.net,'h'+str(i)).weight)))
            self.est_priors['h'+str(i)+'.weight'] = Normal(loc=hw_mu, scale=hw_sigma).to_event(2)
            hb_mu = pyro.param('h'+str(i)+'_b_' + 'mu', torch.randn_like(getattr(self.net,'h'+str(i)).bias))
            hb_sigma = F.softplus(pyro.param('h'+str(i)+'_b_' + 'sigma', torch.randn_like(getattr(self.net,'h'+str(i)).bias)))
            self.est_priors['h'+str(i)+'.bias'] = Normal(loc=hb_mu, scale=hb_sigma).to_event(1)
        
        outw_mu = pyro.param('out'+'_w_' + 'mu', torch.randn_like(self.net.out.weight))
        outw_sigma = F.softplus(pyro.param('out'+'_w_' + 'sigma', torch.randn_like(self.net.out.weight)))
        self.est_priors['out'+'.weight'] = Normal(loc=outw_mu, scale=outw_sigma).to_event(2)
        
        outb_mu = pyro.param('out'+'_b_' + 'mu', torch.randn_like(self.net.out.bias))
        outb_sigma = F.softplus(pyro.param('out'+'_b_' + 'sigma', torch.randn_like(self.net.out.bias)))
        self.est_priors['out'+'.bias'] = Normal(loc=outb_mu, scale=outb_sigma).to_event(1)
        
        # lift module parameters to random variables sampled from the priors
        lifted_module = pyro.random_module("module", self.net, self.est_priors)
        
        return lifted_module()
    
    def Inference(self, X_data, Y_data, loss=Trace_ELBO(), optimizer=Adam, lr=0.01, n_epochs=100, batch_size=128):
        
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
        yhats = [model(X_data).data for model in sampled_models]
        mean = torch.mean(torch.stack(yhats), 0)
        return yhats, torch.argmax(mean, dim=1)
    
class BNN_REG(object):
    
    def __init__(self, n_inputs, n_hiddens=[10], activ_func=F.relu):
        
        super(BNN_REG, self).__init__()
        assert len(n_hiddens) > 0
        
        self.net = NN(n_inputs, n_hiddens, 1, activ_func)
    
    def model(self, features, target):
        
        self.priors = {}
        
        for i in range(len(self.net.hidden_sizes)):
            self.priors['h'+str(i)+'.weight'] = Normal(loc=torch.zeros_like(getattr(self.net,'h'+str(i)).weight), scale=torch.ones_like(getattr(self.net,'h'+str(i)).weight)).to_event(2)
            self.priors['h'+str(i)+'.bias'] = Normal(loc=torch.zeros_like(getattr(self.net,'h'+str(i)).bias), scale=torch.ones_like(getattr(self.net,'h'+str(i)).bias)).to_event(1)
        
        self.priors['out'+'.weight'] = Normal(loc=torch.zeros_like(self.net.out.weight), scale=torch.ones_like(self.net.out.weight)).to_event(2)
        self.priors['out'+'.bias'] = Normal(loc=torch.zeros_like(self.net.out.bias), scale=torch.ones_like(self.net.out.bias)).to_event(1)
        
        # lift module parameters to random variables sampled from the priors
        lifted_module = pyro.random_module("module", self.net, self.priors)
        # sample a regressor (which also samples w and b)
        model_sample = lifted_module()
        
        out_sigma = pyro.sample("sigma", Uniform(0., 10.))
    
        with pyro.plate("data", len(target)):
            
            target_mean = model_sample(features).squeeze(-1)
            pyro.sample("obs", Normal(target_mean, out_sigma), obs=target) #target is not one-hot encoded
            
            return target_mean
    
    def guide(self, features, target):
        
        self.est_priors = {}
        
        for i in range(len(self.net.hidden_sizes)):
            mu = pyro.param('h'+str(i)+'_w_' + 'mu', torch.randn_like(getattr(self.net,'h'+str(i)).weight))
            sigma = F.softplus(pyro.param('h'+str(i)+'_w_' + 'sigma', torch.randn_like(getattr(self.net,'h'+str(i)).weight)))
            self.est_priors['h'+str(i)+'.weight'] = Normal(loc=mu, scale=sigma).to_event(2)
            hb_mu = pyro.param('h'+str(i)+'_b_' + 'mu', torch.randn_like(getattr(self.net,'h'+str(i)).bias))
            hb_sigma = F.softplus(pyro.param('h'+str(i)+'_b_' + 'sigma', torch.randn_like(getattr(self.net,'h'+str(i)).bias)))
            self.est_priors['h'+str(i)+'.bias'] = Normal(loc=hb_mu, scale=hb_sigma).to_event(1)
        
        outw_mu = pyro.param('out'+'_w_' + 'mu', torch.randn_like(self.net.out.weight))
        outw_sigma = F.softplus(pyro.param('out'+'_w_' + 'sigma', torch.randn_like(self.net.out.weight)))
        self.est_priors['out'+'.weight'] = Normal(loc=outw_mu, scale=outw_sigma).to_event(2)
        
        outb_mu = pyro.param('out'+'_b_' + 'mu', torch.randn_like(self.net.out.bias))
        outb_sigma = F.softplus(pyro.param('out'+'_b_' + 'sigma', torch.randn_like(self.net.out.bias)))
        self.est_priors['out'+'.bias'] = Normal(loc=outb_mu, scale=outb_sigma).to_event(1)
        
        # lift module parameters to random variables sampled from the priors
        lifted_module = pyro.random_module("module", self.net, self.est_priors)
        
        sigma_loc = F.softplus(pyro.param('sigma_loc', torch.randn(1)))
        out_sigma = pyro.sample("sigma", Normal(sigma_loc, torch.tensor(0.05)))
        
        return lifted_module()
    
    def Inference(self, X_data, Y_data, loss=Trace_ELBO(), optimizer=Adam, lr=0.01, n_epochs=100, batch_size=128):
        
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
        yhats = [model(X_data).data for model in sampled_models]
        mean = torch.mean(torch.stack(yhats), 0)
        return yhats, mean