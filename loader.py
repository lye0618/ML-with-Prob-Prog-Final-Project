import sys
from zipfile import ZipFile
import sys
import os
import pandas as pd
from common import DATA_DIR, CACHE_DIR


def read_file(data_dir=DATA_DIR, cache_dir=CACHE_DIR, override=False):
    """
    Reads zip and returns content name
    """
    # Check if files already exist in CACHE_DIR and warn user
    if len(os.listdir(cache_dir)) > 1 and not override:
        print(
            "Some files already exist in your CACHE_DIR. If you still want to run this function,run with override=True")
        return
    
    # Extract zip and create msgpack
    for file in os.listdir(data_dir):
        filepath = os.path.join(data_dir, file)
        print(f'filepath = {filepath}')
        with ZipFile(filepath) as zipped:
            filenames = ZipFile.namelist(zipped)
            for name in filenames:
                print(f'file name = {name}')
                cache_path = os.path.join(cache_dir, f'{name}.msgpack')
                with zipped.open(name) as data:
                    df = pd.read_csv(data, na_values=['NaN'])
                    df.drop_duplicates(inplace=True)
                    convert_unix_timestamp(df, 'Timestamp')
                    pd.to_msgpack(cache_path, df)
                    print(f'msgpack saved for {name}')


def read_data(data_type, sample=1):
    """
    Reads in data from cached messagepacks stored locally
    :param data_type: 'bitstamp' or 'coinbase' or 'nyse'
    :param sample: Fraction of randomly sampled data (without replacement) to return. Defaults to full data
    :return: pandas dataframe of requested file
    """
    if data_type == 'bitstamp':
        file_name = 'bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv'
    elif data_type == 'coinbase':
        file_name = 'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
    elif data_type == 'nyse':
        file_name = 'prices-split-adjusted'

    if data_type == 'nyse':
        file_path = os.path.join(DATA_DIR, f'{file_name}.csv')
        df = pd.read_csv(file_path)
        df['return'] = (df['close'] / df['open']) - 1
    else:
        file_path = os.path.join(CACHE_DIR, f'{file_name}.msgpack')
        df = pd.read_msgpack(file_path)

    print(f'Data loaded from {file_path}')

    if sample < 1:
        df_sample = df.sample(frac=sample, random_state=42)
    elif sample == 1:
        df_sample = df
    else:
        sys.exit('Sample size can be atmost 1')

    return df_sample


def convert_unix_timestamp(df, col):
    """
    converts unix timestamp columns to human readable datetime
    :param df: imput dataframe
    :param timecols: list of columns containing unix timestamp
    :return: dataframe with unix timestamp columns converted to datetime
    """

    df[col] = pd.to_datetime(df[col], unit='s', utc=True)


if __name__ == '__main__':
    read_file(DATA_DIR, CACHE_DIR)





