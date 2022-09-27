import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='LD2011_2014.txt')
    parser.add_argument('-o', '--output', type=str, default='ELD.csv')
    args = parser.parse_args()
    data_ELD = pd.read_csv(args.input, parse_dates=True, sep=';', decimal=',', index_col=0)
    data_ELD = data_ELD.resample('1h', closed='right').sum()
    data_ELD = data_ELD.loc[:, data_ELD.cumsum(axis=0).iloc[8920] != 0]  # filter out instances with missing values
    data_ELD.index = data_ELD.index.rename('date')
    data_ELD = data_ELD['2012':]
    data_ELD.to_csv(args.output)
