#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 23:42:44 2019

@author: linye
"""

import pandas as pd

import torch.nn.functional as F

import BNN_helper_upd as BNN
from backtest_helper import Backtest, alpha_generate


def BNN_Network():

    model_data = pd.read_csv('Bayesian_Neural_Network/model_data.csv')
    backtest_data = pd.read_csv('backtest_data.csv')
    model_data['Date'] = pd.to_datetime(model_data['Date'])
    backtest_data['Date'] = pd.to_datetime(backtest_data['Date'])
    rtn_type = 'Rtn1q'
    factors = model_data.columns[3:-1]
    n_inputs = len(factors)
    n_hiddens =[10]
    activ_func = F.relu 
    model = BNN.BNN_REG(n_inputs, n_hiddens, activ_func)
    backtest = Backtest(backtest_data)
    # fake_alpha = backtest_data.rename(columns={'Rtn1q': 'alpha'})
    # backtest.set_alpha(fake_alpha)
    # backtest.print_results()
    alpha_generator = alpha_generate(model_data, factors, rtn_type, model)
    train_begin = '2000-01-01'
    train_end = '2016-12-31'
    test_begin = '2017-01-01'
    test_end = '2019-12-31'
    train_alpha, train_R_2 = alpha_generator.Train(train_begin, train_end)
    backtest.set_alpha(train_alpha)
    backtest.print_results()
    test_alpha, test_R_2 = alpha_generator.Test(test_begin, test_end)
    backtest.set_alpha(test_alpha)
    backtest.print_results()
