# -*- coding: utf-8 -*-
"""
Created on Wed May 11 17:54:22 2022

@author: derre
"""

#%% User inputs
symbol_list_den = ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY', 'EURGBP']
#symbol_list_num = ['USDCAD', 'USDCHF', 'USDJPY']
#compare_symbol_list = ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY']

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
period = '2y'
interval = '1h'


for symbol in symbol_list_den:
    open_daily = yf.download(tickers = symbol + '=X', period  = period, interval = interval)['Close']
    #open_daily = yf.download(tickers = symbol + '=X', interval = interval, start = "2022-04-20", end = "2022-04-25")['Close']
    open_daily = open_daily/open_daily.iloc[0]
    open_daily[open_daily > 2] = float('NaN')
    symbols_df[symbol] = open_daily

symbols_df.ffill()
    
"""
for symbol in symbol_list_num:
    open_daily = yf.download(tickers = symbol + '=X', period  = period, interval = interval)['Close']
    open_daily = open_daily
    open_daily = open_daily/open_daily.iloc[0]
    symbols_df[symbol] = open_daily


symbols_df_wk = pd.DataFrame()
period = '3y'
interval = '1wk'


for symbol in symbol_list_den:
    open_daily = yf.download(tickers = symbol + '=X', period  = period, interval = interval)['Close']
    #open_daily = open_daily.rename(index = {0:symbol})
    open_daily = open_daily/open_daily.iloc[0]
    symbols_df_wk[symbol] = open_daily
    #symbols_df = pd.concat([symbols_df, open_daily], axis = 1)
    

for symbol in symbol_list_num:
    open_daily = yf.download(tickers = symbol + '=X', period  = period, interval = interval)['Close']
    open_daily = open_daily**-1
    open_daily = open_daily/open_daily.iloc[0]
    symbols_df_wk[symbol] = open_daily
    
"""

#%%

symbols_df_save = symbols_df.copy()
#%%
symbols_df = symbols_df_save[['EURUSD']]#, 'GBPUSD']]# 'USDCHF', 'USDJPY']]

#%%
symbols_df = symbols_df_save
#%%
"""
#weekly
roll = 3 # converted to hourlly
shift = 0
diff_threshold = 0.0001
stop = -0.5
diff_range = 26 #converted to hourly

"""

"""
#daily 
roll = 5
shift = 0
diff_threshold = 0.015 #0.01
stop = -0.1

diff_range = 5*20 # 5*25
#diff_range_2 = 5*1 # 5*25

direction = 1

"""
# 1 = curl
# 2 = diff - hourly
# 3 = diff - 1 minute

method = 2

roll = 1
shift = 1
stop = -0.015

if(method == 1):
    #hourly - curl
    
    
    # for curl method
    diff_thresh_dict = {'EURUSD':0.0050,
                        'GBPUSD':0.0065,
                        'AUDUSD':0.0065,
                        'NZDUSD':0.0065,
                        'USDCAD':0.0035,
                        'USDCHF':0.0045,
                        'USDJPY':0.0055,
                        'EURGBP':0.0040}

    
    
    diff_range_dict = { 'EURUSD':24*4,
                        'GBPUSD':24*5,
                        'AUDUSD':24*5,
                        'NZDUSD':24*5,
                        'USDCAD':24*8,
                        'USDCHF':24*2,
                        'USDJPY':2*1,
                        'EURGBP':24*5}
    x = 1
    diff_range_2_dict ={'EURUSD':x*1,
                        'GBPUSD':x*1,
                        'AUDUSD':x*1,
                        'NZDUSD':x*1,
                        'USDCAD':x*1,
                        'USDCHF':x*1,
                        'USDJPY':x*1,
                        'EURGBP':x*1
                        }
    direction = -1
    
if(method == 1.5):
    #hourly - curl
    
    
    # for curl method
    x = 0.002
    diff_thresh_dict = {'EURUSD':x,
                        'GBPUSD':x,
                        'AUDUSD':x,
                        'NZDUSD':x,
                        'USDCAD':x,
                        'USDCHF':x,
                        'USDJPY':x,
                        'EURGBP':x
                        }
    
    
    x = 60
    diff_range_dict = { 'EURUSD':x,
                        'GBPUSD':x,
                        'AUDUSD':x,
                        'NZDUSD':x,
                        'USDCAD':x,
                        'USDCHF':x,
                        'USDJPY':x,
                        'EURGBP':x}
    x = 1
    diff_range_2_dict ={'EURUSD':x*1,
                        'GBPUSD':x*1,
                        'AUDUSD':x*1,
                        'NZDUSD':x*1,
                        'USDCAD':x*1,
                        'USDCHF':x*1,
                        'USDJPY':x*1,
                        'EURGBP':x*1
                        }
    direction = -1

#diff_threshold = 0.01


elif(method == 2):
    # for diff method - hourly
    diff_thresh_dict = {'EURUSD':0.000025,
                        'GBPUSD':0.00002,
                        'AUDUSD':0.00002,
                        'NZDUSD':0.00006,
                        'USDCAD':0.00002,
                        'USDCHF':0.00002,
                        'USDJPY':0.00002,
                        'EURGBP':0.00002}
    
    diff_range_dict =  {'EURUSD':3,
                        'GBPUSD':1,
                        'AUDUSD':1,
                        'NZDUSD':1,
                        'USDCAD':1,
                        'USDCHF':1,
                        'USDJPY':1,
                        'EURGBP':1}
    
    roll_dict =        {'EURUSD':24*7,
                        'GBPUSD':24*4,
                        'AUDUSD':24*4,
                        'NZDUSD':24*4,
                        'USDCAD':24*4,
                        'USDCHF':24*6,
                        'USDJPY':24*6,
                        'EURGBP':24*4}
    direction = -1
    
elif(method == 2.5):
    # for diff method - hourly
    diff_thresh_dict = {'EURUSD':0.0008,
                        'GBPUSD':0.0002,
                        'AUDUSD':0.0002,
                        'NZDUSD':0.0006,
                        'USDCAD':0.0004,
                        'USDCHF':0.0002,
                        'USDJPY':0.0002,
                        'EURGBP':0.0002}
    
    diff_range_dict =  {'EURUSD':1,
                        'GBPUSD':1,
                        'AUDUSD':1,
                        'NZDUSD':1,
                        'USDCAD':1,
                        'USDCHF':1,
                        'USDJPY':1,
                        'EURGBP':1}
    
    roll_dict =        {'EURUSD':8*5,
                        'GBPUSD':24*7,
                        'AUDUSD':24*4,
                        'NZDUSD':24*4,
                        'USDCAD':7*5,
                        'USDCHF':24*6,
                        'USDJPY':24*6,
                        'EURGBP':24*7}
    direction = -1



elif(method == 3):
    # for diff method - 1 minute
    print('diff, 1 minute')
    diff_thresh_dict = {'EURUSD':0.00002,
                        'GBPUSD':0.00002,
                        'AUDUSD':0.00002,
                        'NZDUSD':0.00002,
                        'USDCAD':0.00002,
                        'USDCHF':0.00002,
                        'USDJPY':0.000085,
                        'EURGBP':0.000025}
    
    diff_range_dict =  {'EURUSD':1,
                        'GBPUSD':1,
                        'AUDUSD':1,
                        'NZDUSD':1,
                        'USDCAD':1,
                        'USDCHF':1,
                        'USDJPY':5,
                        'EURGBP':1}
    
    roll_dict =        {'EURUSD':15*6,
                        'GBPUSD':24*4,
                        'AUDUSD':24*4,
                        'NZDUSD':1,
                        'USDCAD':24*4,
                        'USDCHF':24*6,
                        'USDJPY':15*8,
                        'EURGBP':15*5}
    
    direction = -1
    
    


"""
#hourly - standard
roll = 1
shift = 0
stop = -0.05
diff_range = 24*26
#diff_range_2 = 24*26 # 5*25
direction = 1
"""

plot_df = symbols_df.copy()
plot_df_net = symbols_df.copy()
visual = symbols_df.copy()
visual_2 = symbols_df.copy()

if(method < 2):
    # curl
    print('curl')
    for symbol in symbols_df:
        plot_df_net[symbol] = np.sign(symbols_df[symbol].rolling(roll).mean().diff(diff_range_dict[symbol]).diff(diff_range_2_dict[symbol]).shift(shift))
        plot_df_net[symbol][symbols_df[symbol].rolling(roll).mean().diff(diff_range_dict[symbol]).diff(diff_range_2_dict[symbol]).shift(shift).abs() < diff_thresh_dict[symbol]] = float('NaN')
    
else:
    # diff
    for symbol in symbols_df:
        plot_df_net[symbol] = np.sign(symbols_df[symbol].diff(diff_range_dict[symbol]).rolling(roll_dict[symbol]).mean())
        plot_df_net[symbol][symbols_df[symbol].diff(diff_range_dict[symbol]).rolling(roll_dict[symbol]).mean().abs() < diff_thresh_dict[symbol]] = float('NaN')
    



plot_df_net = plot_df_net.ffill()

diff_range = 10 #unknown
diff_range_2 = 10 #unknown
visual = plot_df.rolling(roll).mean().diff(diff_range).shift(shift)
visual_2 = plot_df.rolling(roll).mean().diff(diff_range).diff(diff_range_2).shift(shift)

#plot_df_net = np.sign(plot_df.rolling(roll).mean().diff(diff_range).shift(shift))
#plot_df_net[plot_df.rolling(roll).mean().diff(diff_range).abs() < diff_threshold] = float('NaN')
    




#plot_df_net.iloc[-1] = plot_df_net.iloc[-1]*-1
trades_all = ((plot_df_net.diff()/-2)*plot_df)*direction

trades_net = pd.DataFrame()
sums = pd.DataFrame()
ax = plt.figure()
#ax = symbols_df['GBPUSD'].plot()

for column in trades_all:
    
    trades = trades_all[[column]]
    trades = trades[trades.abs() > 0].dropna()
    trades['add'] = trades[column]
    trades['add'] = trades['add'].shift(1)
    trades = trades[1:].sum(axis = 1)
    trades[trades < stop] = stop
    trades.cumsum().plot()
    trades = pd.DataFrame(trades, columns = ['net'])
    trades['pair'] = column
    #trades[trades.abs() > 0.1] = 0
    
    
    print(column, 'mean =', trades['net'].mean())
    sums = sums.append(pd.DataFrame(trades))


plt.figure()
print(sums['net'].mean())
sums['net'] = sums['net'] - 0.00025
sums = sums.sort_index()
sums['net'].cumsum().plot()
ax.legend()
    

#%%
plot_each = True
time_start = '2022-05'
if(plot_each == True):
    
    for pair in plot_df_net:
        test = plot_df_net[[pair]]/8+1.1
        #test['pair_roll'] = symbols_df[pair].rolling(roll).mean()
        test['pair'] = symbols_df[pair]
        test[time_start:].plot()
    
    ax = plt.figure()
    for pair in plot_df_net:
        test = sums[sums['pair'] == pair]
        test['net'][time_start:].cumsum().plot()


#%%
trades_time = trades[['net']].rename(columns = {'net':'trades'})
trades_time = pd.DataFrame(trades_time, columns = ['trades'])
#trades_time = trades_time.shift(1)
dt = np.diff(trades_time.index)

dt = np.concatenate(([dt.min()], dt))
trades_time['dt'] = dt

dt_win = trades_time[trades_time['trades'] > 0]['dt'].mean()
dt_lose = trades_time[trades_time['trades'] < -0.005]['dt'].min()

print(dt_win)
print(dt_lose)

plt.plot((trades_time[trades_time['trades'] < 0].sort_values('dt')['dt'].values/(1e9*60*60)))
plt.plot((trades_time[trades_time['trades'] > 0].sort_values('dt')['dt'].values/(1e9*60*60)))
#%%
symbols_df = pd.DataFrame()
period = '2y'
interval = '1d'


for symbol in symbol_list_den:
    open_daily = yf.download(tickers = symbol + '=X', period  = period, interval = interval)['Close']
    #open_daily = open_daily.rename(index = {0:symbol})
    #symbols_df = pd.concat([symbols_df, open_daily], axis = 1)
    open_daily = open_daily/open_daily.iloc[0]
    symbols_df[symbol] = open_daily
    

#%%

symbols_df_save = symbols_df.copy()
#%%
symbols_df = symbols_df_save[['EURUSD']]#, 'GBPUSD']]# 'USDCHF', 'USDJPY']]

#%%
symbols_df = symbols_df_save

#%%
#daily - curl
roll = 1
shift = 1
diff_threshold = 0.015
stop = -0.05
diff_range = 5
diff_range_2 = 1 # 5*25
direction = -1

plot_df = symbols_df.copy()
plot_df_net = np.sign(plot_df.rolling(roll).mean().diff(diff_range).diff(diff_range_2).shift(shift))
visual = plot_df.rolling(roll).mean().diff(diff_range).diff(diff_range_2).shift(shift)
plot_df_net[plot_df.rolling(roll).mean().diff(diff_range).diff(diff_range_2).shift(shift).abs() < diff_threshold] = float('NaN')
plot_df_net = plot_df_net.ffill()

plot_df_net.iloc[-1] = plot_df_net.iloc[-1]*-1
trades_all = ((plot_df_net.diff()/-2)*plot_df)*direction

trades_net = pd.DataFrame()
sums = pd.DataFrame()
ax = plt.figure()
#ax = symbols_df['GBPUSD'].plot()

for column in trades_all:
    
    trades = trades_all[[column]]
    trades = trades[trades.abs() > 0].dropna()
    trades['add'] = trades[column]
    trades['add'] = trades['add'].shift(-1)
    trades = trades[:-1].sum(axis = 1)
    trades.cumsum().plot()
    sums = sums.append(pd.DataFrame(trades))
    
print(sums.mean())
sums = sums - 0.00015
sums.sort_index().cumsum().plot()
ax.legend()
    
pair = 'EURUSD'

test = plot_df_net[[pair]]/8+1.1
#test['pair_roll'] = symbols_df[pair].rolling(roll).mean()
test['pair'] = symbols_df[pair]
test.plot()