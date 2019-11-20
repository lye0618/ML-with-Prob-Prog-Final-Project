#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 19:15:36 2019

@author: linye
"""

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt

#Backtest class
class Backtest(object):
    
    def __init__(self, stock_rtn):
        
        self.stock_rtn = stock_rtn
    
    def set_alpha(self, alpha):
        
        self.alpha = alpha
        
    def get_port(self, port_type = 'equal_weighted', neutral='market'):
        
        def alpha_weight(x):
            
            x_demean = x - x.mean()
            return x_demean / np.sum(np.abs(x_demean))
        
        def equal_weight(x):
            
            x_weight = 1.0 * (x - x.quantile(0.8) > 0) - 1.0 * (x - x.quantile(0.2) < 0)
            
            return x_weight / np.sum(np.abs(x_weight))
        
        if port_type == 'equal_weighted':
            if neutral == 'market':
                self.alpha['weights'] = self.alpha.groupby('Date')['alpha'].transform(equal_weight)
            elif neutral == 'industry':
                self.alpha['weights'] = self.alpha.groupby(['Date','Industry'])['alpha'].transform(equal_weight)
            else:
                raise Exception('Not Implemented')
        elif port_type == 'alpha_weighted':  
            if neutral == 'market':
                self.alpha['weights'] = self.alpha.groupby('Date')['alpha'].transform(alpha_weight)
            elif neutral == 'industry':
                self.alpha['weights'] = self.alpha.groupby(['Date','Industry'])['alpha'].transform(alpha_weight)
            else:
                raise Exception('Not Implemented')
        else:
            raise Exception('Not Implemented')
        
    def get_rtns(self, port_type='equal_weighted', neutral='market'):
        
        self.get_port(port_type, neutral)
        
        self.merged_df = pd.merge(self.alpha, self.stock_rtn, how='left', left_on=['Date','Ticker'], right_on = ['Date','Ticker'])
        
        self.merged_df['rtn_times_weight'] = self.merged_df['Rtn1q'] * self.merged_df['weights']
        
        self.port_rtns = self.merged_df.groupby('Date')['rtn_times_weight'].sum()
    
    def summary_statistics(self):
        
        sharpe = np.mean(self.port_rtns) / np.std(self.port_rtns) * np.sqrt(4)
        
        annual_return = np.prod(1+self.port_rtns)**(4/len(self.port_rtns)) - 1
        
        net_values = np.cumprod(1+self.port_rtns)
        
        mdd = np.max(np.maximum.accumulate(net_values) - net_values)
        
        plt.figure(figsize=(10,4))
        plt.plot(self.port_rtns.index, net_values)
        plt.xlabel('Date')
        plt.ylabel('PnL')
        plt.show()
        
        print(f'sharpe ratio {sharpe:.4f}, annual return {annual_return:.4f}, mdd {mdd:.4f}')

        #return ir, sharpe, annual_return, mdd
    
    def print_results(self):
        
        print('alpha weighted, market neutral')
        self.get_rtns(port_type = 'alpha_weighted', neutral='market')
        self.summary_statistics()
        print()
    
        print('equal weighted, market neutral')
        self.get_rtns(port_type = 'equal_weighted', neutral='market')
        self.summary_statistics()
        print()
    
        print('alpha weighted, industry neutral')
        self.get_rtns(port_type = 'alpha_weighted', neutral='industry')
        self.summary_statistics()
        print()
    
        print('equal weighted, industry neutral')
        self.get_rtns(port_type = 'equal_weighted', neutral='industry')
        self.summary_statistics()
        print()

#class to generate alpha
class alpha_generate(object):
    
    def __init__(self, df, factors, rtn_type, model):
        
        self.df = df
        self.factors = factors
        self.rtn_type = rtn_type        
        self.model = model
    
    def Train(self, begin='2016-01-01', end='2017-01-01'):
        
        time_indicator = (self.df['Date'] >= pd.to_datetime(begin, format='%Y-%m-%d')) & (self.df['Date'] < pd.to_datetime(end, format='%Y-%m-%d'))
        
        df = self.df[time_indicator]
        
        X_train = df[self.factors].values
        Y_train = df[self.rtn_type].values
        
        X_tensor = torch.tensor(X_train, dtype=torch.float)
        Y_tensor = torch.tensor(Y_train, dtype=torch.float)
        
        self.model.Inference(X_tensor, Y_tensor, lr=0.001, n_epochs=200, batch_size=128)
        
        df_output = df[['Date','Ticker','Industry']]
        
        df_output['alpha'] = self.model.predict(X_tensor, n_samples=1000)
        
        SSR = np.sum((Y_train - np.mean(Y_train))**2)
        
        SSE = np.sum((Y_train - df_output['alpha'].values)**2)
        
        R_2 = 1 - SSE / SSR
        
        return df_output, R_2
    
    def Test(self, begin, end):
        
        time_indicator = (self.df['Date'] >= pd.to_datetime(begin, format='%Y-%m-%d')) & (self.df['Date'] < pd.to_datetime(end, format='%Y-%m-%d'))
        
        df = self.df[time_indicator]
        
        X_test = df[self.factors].values
        Y_test = df[self.rtn_type].values
        
        X_tensor = torch.tensor(X_test, dtype=torch.float)
        
        df_output = df[['Date','Ticker','Industry']]
        
        df_output['alpha'] = self.model.predict(X_tensor, n_samples=1000)
        
        SSR = np.sum((Y_test - np.mean(Y_test))**2)
        
        SSE = np.sum((Y_test - df_output['alpha'].values)**2)
        
        R_2 = 1 - SSE / SSR
        
        return df_output, R_2