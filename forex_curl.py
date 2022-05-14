# -*- coding: utf-8 -*-
"""
Created on Wed May 11 17:54:22 2022

@author: derre
"""

#%% User inputs
symbol_list_den = ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY', 'EURGBP']
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
period = '2y'
interval = '1h'


for symbol in symbol_list_den:
    open_daily = yf.download(tickers = symbol + '=X', period  = period, interval = interval)['Close']
    #open_daily = open_daily.rename(index = {0:symbol})
    #symbols_df = pd.concat([symbols_df, open_daily], axis = 1)
    open_daily = open_daily/open_daily.iloc[0]
    symbols_df[symbol] = open_daily
    
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
#hourly - curl
roll = 1
shift = 1
diff_threshold = 0.0065
diff_thresh_dict = {'EURUSD':0.0065,
                    'GBPUSD':0.0065,
                    'AUDUSD':0.0065,
                    'NZDUSD':0.0065,
                    'USDCAD':0.0035,
                    'USDCHF':0.0045,
                    'USDJPY':0.0055,
                    'EURGBP':0.0040}
"""
{'EURUSD':0.003,
                    'GBPUSD':0.0065,
                    'AUDUSD':0.0065,
                    'NZDUSD':0.0065,
                    'USDCAD':0.0035,
                    'USDCHF':0.0065,
                    'USDJPY':0.0065}

"""

stop = -0.15
diff_range = 24*5
diff_range_2 = 1*1 # 5*25
diff_range_dict = {'EURUSD':24*10,
                    'GBPUSD':24*5,
                    'AUDUSD':24*5,
                    'NZDUSD':24*5,
                    'USDCAD':24*8,
                    'USDCHF':24*2,
                    'USDJPY':24*10,
                    'EURGBP':24*5}

diff_range_2_dict = {'EURUSD':3*1,
                    'GBPUSD':1*1,
                    'AUDUSD':1*1,
                    'NZDUSD':1*1,
                    'USDCAD':1*1,
                    'USDCHF':1*1,
                    'USDJPY':1*1,
                    'EURGBP':1*1
                    }


direction = 1

"""
#hourly - standard
roll = 1
shift = 0
diff_threshold = 0.01
stop = -0.05
diff_range = 24*26
#diff_range_2 = 24*26 # 5*25
direction = 1
"""

plot_df = symbols_df.copy()
plot_df_net = symbols_df.copy()
visual = symbols_df.copy()
visual_2 = symbols_df.copy()


for symbol in symbols_df:
    plot_df_net[symbol] = np.sign(symbols_df[symbol].rolling(roll).mean().diff(diff_range_dict[symbol]).diff(diff_range_2_dict[symbol]).shift(shift))
    plot_df_net[symbol][symbols_df[symbol].rolling(roll).mean().diff(diff_range_dict[symbol]).diff(diff_range_2_dict[symbol]).shift(shift).abs() < diff_thresh_dict[symbol]] = float('NaN')


plot_df_net = plot_df_net.ffill()
visual = plot_df.rolling(roll).mean().diff(diff_range).shift(shift)
visual_2 = plot_df.rolling(roll).mean().diff(diff_range).diff(diff_range_2).shift(shift)

#plot_df_net = np.sign(plot_df.rolling(roll).mean().diff(diff_range).shift(shift))
#plot_df_net[plot_df.rolling(roll).mean().diff(diff_range).abs() < diff_threshold] = float('NaN')
    










"""
plot_df = symbols_df.copy()
plot_df_net = np.sign(plot_df.rolling(roll).mean().diff(diff_range).diff(diff_range_2).shift(shift))
visual = plot_df.rolling(roll).mean().diff(diff_range).diff(diff_range_2).shift(shift)
plot_df_net[plot_df.rolling(roll).mean().diff(diff_range).diff(diff_range_2).shift(shift).abs() < diff_threshold] = float('NaN')



plot_df_net = plot_df_net.ffill()
"""



#plot_df_net = plot_df_net  + 22
#ratio = (plot_df_net.mean()/plot_df_net.min())/(plot_df['Open'].mean()/plot_df['Open'].min())
#print(ratio)
#plot_df_net = plot_df_net/plot_df_net.max()*plot_df['Open'].max()

#ax3 = ax1.twinx()
#ax3.spines['right'].set_position(('axes', 1.1))
#plot_df['net'].plot(ax = ax3, color = 'b')
#(plot_df['CPI_Energy_ratio']*11).plot()
#((plot_df['CPI_Energy_ratio']*6+ plot_df['assets_per_gdp_diff']*-1 + plot_df['rate_dif'].loc[year:]*1 + plot_df['cpi_ratio']*3+plot_df['m3_ratio']*0)).plot()
#(0.15*plot_df['cpi_ratio']+1).plot()
#((assets_per_gdp_ratio).diff()*100+1).plot()
#(assets_per_gdp_diff.diff()+1).plot(ax = ax)
#(plot_df['assets_per_gdp_diff']+1).plot()

plot_df_net.iloc[-1] = plot_df_net.iloc[-1]*-1
trades_all = ((plot_df_net.diff()/2)*plot_df)*direction

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
    trades[trades < stop] = stop
    trades.cumsum().plot()
    sums = sums.append(pd.DataFrame(trades))


    """
        
    net_column = pd.DataFrame(net_column.rename('net'))
    trades_net = trades_net.append(net_column)
    """
print(sums.mean())
sums = sums - 0.00015
sums.sort_index().cumsum().plot()
ax.legend()
    
    
"""
trades_net.sort_index().cumsum().plot()
print(sums.sum())
print(trades_net.sum())
"""
pair = 'EURUSD'
test = plot_df_net[[pair]]/8+1.1
#test['pair_roll'] = symbols_df[pair].rolling(roll).mean()
test['pair'] = symbols_df[pair]
test.plot()



"""
trades_net = pd.concat([short, long], axis = 1, ignore_index = True)



#trades_net = trades_net.set_index(new_index)
trades_net = trades_net.dropna().sum(axis = 1)
trades_net[trades_net < stop] = stop
total = trades_net.sum()
print(total)
#trades_net.cumsum().plot()
plt.figure()
trades_net.cumsum().plot()

"""
#%%

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