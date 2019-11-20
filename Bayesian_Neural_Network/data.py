#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:57:31 2019

@author: linye
"""

import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def preprocess():
  data = pd.read_csv("data_fund.csv")
  fun = data.loc[~data['Indicator Name'].isin(['Common Shares Outstanding','Share Price'])].reset_index(drop=True)
  fun['yyyymm']=pd.to_datetime(fun['publish date'])
  
  fun['yearmonth'] =  fun['yyyymm'].map(lambda x: 100*x.year + x.month)
  fun = fun.groupby(['Ticker','yearmonth','Indicator Name','SimFin ID','Company Industry Classification Code'])['Indicator Value'].mean().reset_index()
  fun['yyyymm'] = fun['yearmonth'].map(lambda x:datetime(int(str(x)[:4]), int(str(x)[4:6]), 1)+ relativedelta(months=1)-timedelta(days=1))
  fun['yyyymm'] = pd.to_datetime(fun['yyyymm'])
  
  
  fun = pd.pivot_table(fun, values='Indicator Value', index=['Ticker', 'yyyymm'],columns=['Indicator Name'], aggfunc=np.mean).reset_index()

  to_remove = ['Ticker','yyyymm','Avg. Basic Shares Outstanding','Avg. Diluted Shares Outstanding','Total Assets']
  to_remove2 = ['COGS', 'EBIT', 'Cash From Operating Activities', 'Cash and Cash Equivalents',
               'Net Profit', 'Equity Before Minorities', 'Total Noncurrent Liabilities', 'Total Equity',
               'Total Noncurrent Assets', 'Retained Earnings','Net Income from Discontinued Op.','Preferred Equity','Share Capital','Intangible Assets']
  to_remove = to_remove + to_remove2
  fun_cols = list(fun.drop(columns=to_remove).columns)
  # for elem in to_remove:
  #   fun_cols.remove(elem)
  print(fun_cols)
  fun[fun_cols] = fun[fun_cols].div(fun['Total Assets'], axis=0).reset_index(drop=True)


  features = fun[fun_cols+['yyyymm','Ticker']]
  features = features.dropna().reset_index(drop=True)
  features = features.fillna(0).reset_index(drop=True)

  zscore = lambda x: (x - x.mean()) / x.std()
  final = features[['Ticker','yyyymm']]
  for item in fun_cols:
    temp = features.groupby([features.yyyymm])[item].transform(zscore)
    final = pd.concat([final,temp], axis=1).reset_index(drop=True)

  data['publish date']=pd.to_datetime(data['publish date'])
  rtns = data.loc[data['Indicator Name']=='Share Price', ['Ticker','publish date','Company Industry Classification Code', 'Indicator Value']].reset_index(drop=True)
  rtns.columns = ['Ticker','Date','Industry','Price']
  rtns['Date'] = pd.to_datetime(rtns['Date'])
  #rtns_new = rtns.groupby(['Date','Ticker'])['Industry','Price'].apply(lambda x: x.mean()).reset_index()
  # rtns['Rtn1d'] = rtns.groupby('Ticker')['Price'].apply(lambda x: np.log(x).diff())
  # rtns['Rtn1q'] = rtns.groupby('Ticker')['Rtn1d'].rolling(63).sum().reset_index(0,drop=True)
  #rtns['Rtn1d'] = rtns.groupby('Ticker')['Price'].apply(lambda x: np.log(x).diff())
  #rtns['pmom'] = rtns.groupby('Ticker')['Rtn1d'].rolling(63).sum().reset_index(0,drop=True)
  #rtns['Rtn1q'] = rtns.groupby('Ticker')['Rtn1d'].rolling(63).sum().shift(-63).reset_index(0,drop=True)
  #rtns['Rtn1d'] = rtns.groupby('Ticker')['Rtn1d'].shift(-1).reset_index(0,drop=True)
  #rtns.drop(columns=['Price','Rtn1d'], inplace=True)
  #rtns.dropna(inplace=True)

  final = final.merge(rtns, left_on=['Ticker','yyyymm'], right_on =['Ticker','Date'],how='inner').reset_index(0,drop=True)
  final.dropna(inplace=True)
  
  final.drop_duplicates(subset=['Ticker','Date'], keep='last', inplace=True)
  
  final['Prtn1q'] = final.groupby('Ticker')['Price'].apply(lambda x: np.log(x).diff())
  final['Rtn1q'] = final.groupby('Ticker')['Prtn1q'].shift(-1)
  #final['Normal_Rtn1q'] = final.groupby('Date')['Price'].apply(lambda x: np.log(x).diff())
  
  final.dropna(inplace=True)
  
  #final['Prtn1q'] = final.groupby('Date')['Prtn1q'].transform(lambda x: (x - x.mean()) / x.std())
  #final['Normal_Rtn1q'] = final.groupby('Ticker')['Prtn1q'].shift(-1)
  
  #final.dropna(inplace=True)
  
  #final.set_index('Date',inplace=True)
  
  model_cols = ['Date','Ticker','Industry'] + fun_cols + ['Prtn1q','Rtn1q']
  backtest_cols = ['Date','Ticker','Industry','Rtn1q']
  
  return final, final[model_cols], final[backtest_cols]


final, model_data, backtest_data = preprocess()

model_data.to_csv('model_data.csv',index=False)
backtest_data.to_csv('backtest_data.csv',index=False)