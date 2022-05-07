# -*- coding: utf-8 -*-
"""
Created on Wed May  4 12:35:30 2022

@author: derre
"""
#%% User inputs
symbol_list_den = ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD']
symbol_list_num = ['USDCAD', 'USDCHF', 'USDJPY']
compare_symbol_list = ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY']

#%%Import standard modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import plotly.graph_objs as go

# import yahoo finance data
import yfinance as yf
# example: data_daily = yf.download(tickers = 'EURUSD=X', period  = '3y', interval = '1d')

# Import FRED data: Federal Reserve Economic Data
import fredapi as fa
#example: fred = fa.Fred(api_key = '47aed364aa18b64e34f2f9695bd6dd67')

#%%

symbols_df = pd.DataFrame()
period = '20d'
interval = '5m'


for symbol in symbol_list_den:
    open_daily = yf.download(tickers = symbol + '=X', period  = period, interval = interval)['Close']
    #open_daily = open_daily.rename(index = {0:symbol})
    #symbols_df = pd.concat([symbols_df, open_daily], axis = 1)
    open_daily = open_daily/open_daily.iloc[0]
    symbols_df[symbol] = open_daily
    

for symbol in symbol_list_num:
    open_daily = yf.download(tickers = symbol + '=X', period  = period, interval = interval)['Close']
    open_daily = open_daily**-1
    open_daily = open_daily/open_daily.iloc[0]
    symbols_df[symbol] = open_daily

#%% comparison
#single_symbol = symbols_df[[compare_symbol]]
#single_symbol = yf.download(tickers = compare_symbol + '=X', period  = period, interval = interval)['Open']
#single_symbol_2 = yf.download(tickers = compare_symbol_2 + '=X', period  = period, interval = interval)['Open']
compare = pd.DataFrame(symbols_df.mean(axis = 1), columns = ['mean']).iloc[:]
compare = compare/compare.iloc[0]


for symbol in compare_symbol_list:
    compare[symbol] = (symbols_df[[symbol]]/symbols_df[[symbol]].iloc[0]).iloc[:]

#compare[compare_symbol_2] = single_symbol_2/single_symbol_2.iloc[0]
compare.plot()

#%%
start = 14
delta = 325
end = start + delta
compare_slice = compare.iloc[start:end]
#compare_slice = compare.copy()

compare_slice = compare_slice/compare_slice.iloc[0]
#compare_slice.diff(axis = 1).plot()
compare_slice.plot()

#%%
compare.iloc[:,0].subtract(compare.iloc[:,1]).plot()