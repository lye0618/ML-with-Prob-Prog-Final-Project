import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


def preprocess(ret_type):
    data = pd.read_csv("/mnt/d/mlpp/nyse/data_fund.csv")
    fun = data.loc[~data['Indicator Name'].isin(
        ['Common Shares Outstanding', 'Share Price'])].reset_index(drop=True)

    fun['yyyymm'] = pd.to_datetime(fun['publish date'])

    fun['yearmonth'] = fun['yyyymm'].map(lambda x: 100*x.year + x.month)

    cols = ['Ticker', 'yearmonth', 'Indicator Name', 'SimFin ID',
            'Company Industry Classification Code']

    fun = fun.groupby(cols)['Indicator Value'].mean().reset_index()

    fun['yyyymm'] = fun['yearmonth'].map(
        lambda x: datetime(int(str(x)[:4]),
                           int(str(x)[4:6]), 1
                           ) + relativedelta(months=1) - timedelta(days=1))

    fun['yyyymm'] = pd.to_datetime(fun['yyyymm'])

    fun = pd.pivot_table(fun,
                         values='Indicator Value',
                         index=['Ticker', 'yyyymm'],
                         columns=['Indicator Name'],
                         aggfunc=np.mean).reset_index()

    to_remove = ['Ticker', 'yyyymm', 'Avg. Basic Shares Outstanding',
                 'Avg. Diluted Shares Outstanding', 'Total Assets']

    to_remove2 = ['COGS', 'EBIT',
                  'Cash From Operating Activities',
                  'Cash and Cash Equivalents',
                  'Net Profit', 'Equity Before Minorities',
                  'Total Noncurrent Liabilities',
                  'Total Equity',
                  'Total Noncurrent Assets', 'Retained Earnings',
                  'Net Income from Discontinued Op.', 'Preferred Equity',
                  'Share Capital', 'Intangible Assets']

    to_remove = to_remove + to_remove2
    fun_cols = list(fun.drop(columns=to_remove).columns)
    # for elem in to_remove:
    #   fun_cols.remove(elem)

    fun[fun_cols] = fun[fun_cols].div(
        fun['Total Assets'], axis=0).reset_index(drop=True)

    features = fun[fun_cols+['yyyymm', 'Ticker']]
    features = features.dropna().reset_index(drop=True)
    features = features.fillna(0).reset_index(drop=True)

    def zscore(x):
        return (x - x.mean()) / x.std()

    final = features[['Ticker', 'yyyymm']]

    for item in fun_cols:
        temp = features.groupby([features.yyyymm])[item].transform(zscore)
        final = pd.concat([final, temp], axis=1).reset_index(drop=True)

    data['publish date'] = pd.to_datetime(data['publish date'])
    rtns = data.loc[data['Indicator Name'] == 'Share Price',
                    ['Ticker', 'publish date',
                     'Company Industry Classification Code',
                     'Indicator Value']].reset_index(drop=True)

    rtns.columns = ['Ticker', 'Date', 'Industry', 'Price']

    rtns = rtns.groupby(['Ticker', 'Date', 'Industry'])['Price']\
        .mean().reset_index()

    rtns['Rtn1d'] = rtns.groupby('Ticker')['Price']\
        .apply(lambda x: np.log(x).diff())
    rtns['pmom'] = rtns.groupby('Ticker')['Rtn1d'].rolling(63)\
        .sum().reset_index(0, drop=True)
    rtns['Rtn1q'] = rtns.groupby('Ticker')['Rtn1d'].rolling(63)\
        .sum().shift(-63).reset_index(0, drop=True)
    rtns['Rtn1d'] = rtns.groupby('Ticker')['Rtn1d'].shift(-1)\
        .reset_index(0, drop=True)

    rtns.dropna(inplace=True)

    final = final.merge(rtns,
                        left_on=['Ticker', 'yyyymm'],
                        right_on=['Ticker', 'Date'],
                        how='inner').reset_index(0, drop=True)

    final.dropna(inplace=True)
    train = final[final['yyyymm'] <= '2016-12-31'].reset_index(0, drop=True)
    test = final[final['yyyymm'] > '2016-12-31'].reset_index(0, drop=True)
    fun_cols = fun_cols + ['pmom']

    if ret_type == 'tensor':
        x_train = train[fun_cols].values
        y_train = train['Rtn1q'].values
        x_test = test[fun_cols].values
        y_test = test['Rtn1q'].values
        x_tr = torch.tensor(x_train, dtype=torch.float)
        y_tr = torch.tensor(y_train, dtype=torch.float)
        x_ts = torch.tensor(x_test, dtype=torch.float)
        y_ts = torch.tensor(y_test, dtype=torch.float)

    elif ret_type == 'df':
        x_tr = train[fun_cols + ['yyyymm', 'Ticker']]
        y_tr = train['Rtn1q']
        x_ts = test[fun_cols + ['yyyymm', 'Ticker']]
        y_ts = test['Rtn1q']

    return x_tr, y_tr, x_ts, y_ts
