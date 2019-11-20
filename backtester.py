import numpy as np
import pandas as pd
import torch
from backtest_helper import Backtest


def merge_with_tix(resps, lin_preds, x_test):
    # GMM responsibilities
    resps_np = resps.numpy()
    secs = np.argmax(resps_np, axis=1)
    secs = pd.Series(secs)
    secs.rename('Industry', inplace=True)
    x = pd.concat([x_test, secs], axis=1)
    x = x.rename(columns={'yyyymm': 'Date'})
    y = x[['Date', 'Ticker', 'Industry']]

    # Linear Regr Predictions
    p = lin_preds.squeeze()
    p = p.numpy()
    preds2 = pd.Series(p)
    preds2.rename('alpha', inplace=True)
    final = pd.concat([y, preds2], axis=1)
    assert final.shape == (x_test.shape[0], 4)
    return final


def get_data():
    backtest_data = pd.read_csv('/mnt/d/mlpp/nyse/backtest_data.csv')
    backtest_data['Date'] = pd.to_datetime(backtest_data['Date'])
    return backtest_data


def main(final):
    backtest = Backtest(get_data())
    backtest.set_alpha(final)
    backtest.print_results()

