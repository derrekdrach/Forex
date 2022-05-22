# -*- coding: utf-8 -*-
"""
Created on Wed May 11 17:54:22 2022

@author: derre
"""


#%% User inputs
symbol_list = ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY', 'EURGBP']
futures_list = ['ZR']
#symbol_list_den = ['EURUSD', 'AUDUSD']
#compare_symbol_list = ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY']

#Import standard modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# import yahoo finance data
import yfinance as yf
# example: data_daily = yf.download(tickers = 'EURUSD=X', period  = '3y', interval = '1d')

# Import FRED data: Federal Reserve Economic Data
#import fredapi as fa
#example: fred = fa.Fred(api_key = '47aed364aa18b64e34f2f9695bd6dd67')
#%% commodity futures



#%%

symbols_df = pd.DataFrame()
period = '2y'
interval = '1h'


for symbol in symbol_list:
    print(symbol)

    open_df = yf.download(tickers = symbol + '=X', period  = period, interval = interval)['Close']
    #open_daily = yf.download(tickers = symbol + '=X', interval = interval, start = "2022-04-20", end = "2022-04-25")['Close']
    symbols_df[symbol] = open_df
    #open_df = open_df/open_df.iloc[0]
    #open_df[open_df > 2] = float('NaN')
    


symbols_df[symbols_df > 2] = float('NaN')    
symbols_df_price = symbols_df.copy()
symbols_df = symbols_df_price/symbols_df_price.iloc[0]
symbols_df = symbols_df.ffill()

symbols_df_dly = symbols_df.resample('d').first().dropna()

#%%
symbols_df_dly = pd.DataFrame()
period = '16y'
interval = '1d'


for symbol in symbol_list:
    open_df = yf.download(tickers = symbol + '=X', period  = period, interval = interval)['Close']    
    symbols_df_dly[symbol] = open_df


symbols_df_dly_price = symbols_df_dly.copy()
symbols_df_dly = symbols_df_dly_price/symbols_df_dly_price.iloc[0]
symbols_df_dly[symbols_df_dly > 2] = float('NaN')
symbols_df_dly = symbols_df_dly.ffill()
    
#%%
symbol_list_den = ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY', 'EURGBP']
#symbol_list_den = ['EURUSD', 'GBPUSD']

symbols_df = pd.DataFrame()


file_directory = r'C:\Users\derre\Desktop\data\scripts'
for symbol in symbol_list_den:
    file_name = '_dataout.txt'
    open_df = pd.read_csv(file_directory + '\\' +  symbol + file_name, names = [symbol])
    file_name = '_timeStamp.txt'
    open_df.index = pd.read_csv(file_directory + '\\' + symbol + file_name, header = None).values.flatten()
    symbols_df[symbol] = open_df[symbol]
symbols_df = symbols_df/symbols_df.iloc[-1]
symbols_df.index = pd.to_datetime(symbols_df.index)
symbols_df_dly = symbols_df.resample('d').first().dropna()
#%%

symbols_df_dly_save = symbols_df_dly.copy()
symbols_df_save = symbols_df.copy()

#%%
symbols_df_dly = symbols_df_dly_save[['EURUSD', 'E']]#, 'GBPUSD']]# 'USDCHF', 'USDJPY']]

symbols_df = symbols_df_save[['EURUSD']]




#%%
symbols_df_dly = symbols_df_dly_save

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


roll = 1
shift = 1
#stop = -0.025



# for diff method - daily
diff_thresh_dict = {'EURUSD':0.0008,
                    'GBPUSD':0.0008,
                    'AUDUSD':0.0008,
                    'NZDUSD':0.0012,
                    'USDCAD':0.001,
                    'USDCHF':0.0008,
                    'USDJPY':0.0008,
                    'EURGBP':0.0008}

diff_range_dict =  {'EURUSD':1,
                    'GBPUSD':1,
                    'AUDUSD':1,
                    'NZDUSD':1,
                    'USDCAD':1,
                    'USDCHF':1,
                    'USDJPY':1,
                    'EURGBP':1}

roll_dict =        {'EURUSD':8*5,
                    'GBPUSD':8*5,
                    'AUDUSD':8*5,
                    'NZDUSD':8*5,
                    'USDCAD':6*5,
                    'USDCHF':7*5,
                    'USDJPY':9*5,
                    'EURGBP':6*5}
direction = 1


plot_df_net_loop = symbols_df_dly.copy()
symbols_loop_df = symbols_df_dly.copy()
for symbol in symbols_loop_df:
    plot_df_net_loop[symbol] = np.sign(symbols_loop_df[symbol].diff(diff_range_dict[symbol]).rolling(roll_dict[symbol]).mean())
    plot_df_net_loop[symbol][symbols_loop_df[symbol].diff(diff_range_dict[symbol]).rolling(roll_dict[symbol]).mean().abs() < diff_thresh_dict[symbol]] = float('NaN')
plot_df_net_loop = plot_df_net_loop.ffill()    
plot_df_net_dly = plot_df_net_loop.copy()
"""
diff_thresh_dict = {'EURUSD':0.03,
                    'GBPUSD':0.03,
                    'AUDUSD':0.03,
                    'NZDUSD':0.03,
                    'USDCAD':0.03,
                    'USDCHF':0.03,
                    'USDJPY':0.03,
                    'EURGBP':0.03}

x = 5*25
diff_range_dict =  {'EURUSD':x,
                    'GBPUSD':x,
                    'AUDUSD':x,
                    'NZDUSD':x,
                    'USDCAD':x,
                    'USDCHF':x,
                    'USDJPY':x,
                    'EURGBP':x}


diff_range_2_dict ={'EURUSD':x*1,
                    'GBPUSD':x*1,
                    'AUDUSD':x*1,
                    'NZDUSD':x*1,
                    'USDCAD':x*1,
                    'USDCHF':x*1,
                    'USDJPY':x*1,
                    'EURGBP':x*1
                    }

roll_dict =        {'EURUSD':8*5,
                    'GBPUSD':8*5,
                    'AUDUSD':8*5,
                    'NZDUSD':8*5,
                    'USDCAD':6*5,
                    'USDCHF':7*5,
                    'USDJPY':9*5,
                    'EURGBP':6*5}
direction = 1


plot_df_net_loop = symbols_df_dly.copy()
symbols_loop_df = symbols_df_dly.copy()
for symbol in symbols_loop_df:
    plot_df_net_loop[symbol] = np.sign(symbols_loop_df[symbol].rolling(roll).mean().diff(diff_range_dict[symbol]).diff(diff_range_2_dict[symbol]).shift(shift))
    plot_df_net_loop[symbol][symbols_loop_df[symbol].rolling(roll).mean().diff(diff_range_dict[symbol]).diff(diff_range_2_dict[symbol]).shift(shift).abs() < diff_thresh_dict[symbol]] = float('NaN')
    
    
plot_df_net_loop = plot_df_net_loop.ffill()    
plot_df_net_dly = plot_df_net_loop.copy()
"""

method = 2

if(method == 1):
# for curl method - hourly
    diff_thresh_dict = {'EURUSD':0.0050,
                        'GBPUSD':0.0065,
                        'AUDUSD':0.0065,
                        'NZDUSD':0.0065,
                        'USDCAD':0.0035,
                        'USDCHF':0.0035,
                        'USDJPY':0.0055,
                        'EURGBP':0.0050,
                        'EURGBP':0.0050}
    
    
    
    diff_range_dict = { 'EURUSD':24*4,
                        'GBPUSD':24*1,
                        'AUDUSD':24*5,
                        'NZDUSD':24*5,
                        'USDCAD':24*8,
                        'USDCHF':24*8,
                        'USDJPY':2*1,
                        'EURGBP':24*5}
    
    diff_range_dict = { 'EURUSD':24*4,
                        'GBPUSD':24*1,
                        'AUDUSD':24*5,
                        'NZDUSD':24*5,
                        'USDCAD':24*8,
                        'USDCHF':24*10,
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

elif(method == 2):
    # for diff method - hourly
    diff_thresh_dict = {'EURUSD':0.00005,
                        'GBPUSD':0.00003,
                        'AUDUSD':0.00006,
                        'NZDUSD':0.0001,
                        'USDCAD':0.00002,
                        'USDCHF':0.00002,
                        'USDJPY':0.00001,
                        'EURGBP':0.00002}
    
    diff_range_dict =  {'EURUSD':1,
                        'GBPUSD':1,
                        'AUDUSD':1,
                        'NZDUSD':1,
                        'USDCAD':1,
                        'USDCHF':1,
                        'USDJPY':1,
                        'EURGBP':1}
    
    roll_dict =        {'EURUSD':24*15,
                        'GBPUSD':24*10,
                        'AUDUSD':24*4,
                        'NZDUSD':24*6,
                        'USDCAD':24*5,
                        'USDCHF':24*15,
                        'USDJPY':24*15,
                        'EURGBP':24*10}
    
    #x = 24*10
    #roll_dict =        {'EURUSD':x,
    #                    'GBPUSD':x,
    #                    'AUDUSD':24*15,
    #                    'NZDUSD':24*10,
    #                    'USDCAD':24*10,
    #                    'USDCHF':24*10,
    #                    'USDJPY':24*10,
    #                    'EURGBP':24*10}
    
    

    direction = -1


plot_df_net_loop = symbols_df.copy()
symbols_loop_df = symbols_df.copy()

if(method < 2):
    # curl
    print('curl')
    for symbol in symbols_df:
        plot_df_net_loop[symbol] = np.sign(symbols_loop_df[symbol].rolling(roll).mean().diff(diff_range_dict[symbol]).diff(diff_range_2_dict[symbol]).shift(shift))
        plot_df_net_loop[symbol][symbols_loop_df[symbol].rolling(roll).mean().diff(diff_range_dict[symbol]).diff(diff_range_2_dict[symbol]).shift(shift).abs() < diff_thresh_dict[symbol]] = float('NaN')
    
else:
    # diff
    for symbol in symbols_df:
        plot_df_net_loop[symbol] = np.sign(symbols_loop_df[symbol].diff(diff_range_dict[symbol]).rolling(roll_dict[symbol]).mean())
        plot_df_net_loop[symbol][symbols_loop_df[symbol].diff(diff_range_dict[symbol]).rolling(roll_dict[symbol]).mean().abs() < diff_thresh_dict[symbol]] = float('NaN')



plot_df_net_loop = plot_df_net_loop.ffill()
plot_df_net = plot_df_net_loop.copy()
plot_df_net = plot_df_net*direction


plot_df_net_dly_on_hr = plot_df_net_dly.resample('H').ffill()
#plot_df_net_sum = plot_df_net_dly_on_hr.add(plot_df_net)
plot_df_net_events = np.sign(plot_df_net_dly_on_hr.add(plot_df_net).abs().ffill().diff())
#plot_df_net_events = np.sign(plot_df_net.diff())


plot_df_net_entries = pd.DataFrame()
plot_df_net_exits = pd.DataFrame()
for symbol in symbols_df:    
    plot_df_net_entries = pd.concat([plot_df_net_entries, plot_df_net[symbol][plot_df_net_events[symbol] > 0]], axis = 1)
    plot_df_net_exits = pd.concat([plot_df_net_exits, plot_df_net[symbol][plot_df_net_events[symbol] < 0]], axis = 1)
    
        
    

stop = -0.5
limit = 0.15
sums = pd.DataFrame()

for symbol in symbols_df:
        
    
    symbol_entries = plot_df_net_entries[symbol].dropna()
    symbol_entries = -1*symbol_entries*symbols_df[symbol].loc[symbol_entries.index]
    
    
    symbol_exits = plot_df_net_exits[symbol].dropna()
    symbol_exits = symbol_exits[symbol_exits.index > symbol_entries.index[0]]
    if(symbol_entries.index[-1] > symbol_exits.index[-1]):
        symbol_exits = pd.concat([symbol_exits, (symbols_df.iloc[[-1]][symbol]*0+1)])
    symbol_exits = symbol_exits*symbols_df[symbol].loc[symbol_exits.index]
    
    
    
        
    
    
    
    symbol_net = pd.DataFrame(symbol_entries.add((symbol_exits.abs()*-1*np.sign(symbol_entries).values).values))
    symbol_net = symbol_net.rename(columns = {symbol:'net'})
    
    for index, item in enumerate(symbol_entries):   
        
        stop_flag = 0
        
        
        
        
        if(symbol_entries[index] < 0):
            diff_open = symbols_df[symbol].loc[symbol_entries.index[index]:symbol_exits.index[index]].min() + symbol_entries[index]
            if(np.abs(diff_open) > np.abs(stop)):
                print('stop: ', diff_open)
                symbol_net['net'].iloc[index] = stop
                stop_flag = 1
                
        if(symbol_entries[index] > 0):
            diff_open = symbols_df[symbol].loc[symbol_entries.index[index]:symbol_exits.index[index]].max() - symbol_entries[index]
            if(np.abs(diff_open) > np.abs(stop)):
                print('stop: ', diff_open)
                symbol_net['net'].iloc[index] = stop
                stop_flag = 1
                
        if(stop_flag == 0):     
            if(symbol_entries[index] < 0):
                diff_open = symbols_df[symbol].loc[symbol_entries.index[index]:symbol_exits.index[index]].max() + symbol_entries[index]
                if(np.abs(diff_open) > np.abs(limit)):
                    print('limit: ', diff_open)
                    symbol_net['net'].iloc[index] = limit
            
            
            if(symbol_entries[index] > 0):
                diff_open = symbols_df[symbol].loc[symbol_entries.index[index]:symbol_exits.index[index]].min() - symbol_entries[index]
                if(np.abs(diff_open) > np.abs(limit)):
                    print('limit: ', diff_open)
                    symbol_net['net'].iloc[index] = limit
                    
            
                
                

                
                
    #symbol_net[symbol_net < stop] = stop
    print(symbol, "sum", symbol_net.sum().values)
    symbol_net['pair'] = symbol
    sums = sums.append(symbol_net)
    

#sums[sums['net'] < stop]['net'] = stop
sums = sums.sort_index()
sums['net'] = sums['net'] - 0.00025
sums['net'].cumsum().plot()
print('sum = ', sums['net'].sum())
print('mean = ', sums['net'].mean())








plot_df_net_plot = plot_df_net[plot_df_net_dly_on_hr.add(plot_df_net).abs() > 0].fillna(0)

plot_each = True
time_start = '2000-03'
if(plot_each == True):
    
    for pair in plot_df_net_plot:
        test = plot_df_net_plot[[pair]]/8+1.1
        #test['pair_roll'] = symbols_df[pair].rolling(roll).mean()
        test['pair'] = symbols_df[pair]
        test[time_start:].plot()
    
ax = plt.figure()
for pair in plot_df_net:
    test = sums[sums['pair'] == pair]
    test['net'][time_start:].cumsum().plot()

ax.legend()
    


#%%
####################################


######################################
#%% Old






method = 2


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
    

#%%

#%%
stop = -0.25
direction = -1

plot_df_net = plot_df_net.copy()
plot_df = symbols_df.copy()

plot_df_net.iloc[-1] = plot_df_net.iloc[-1]*-1
trades_all = plot_df_net.diff()*-0.5*symbols_df*direction

trades_net = pd.DataFrame()
sums = pd.DataFrame()
ax = plt.figure()
#ax = symbols_df['GBPUSD'].plot()

for column in trades_all:
    trades = trades_all[[column]].copy()
    trades = trades[trades.abs() > 0].dropna()
    trades['add'] = trades[column]
    trades['add'] = trades['add'].shift(1)
    trades = trades[1:].sum(axis = 1)
    #trades = trades/1
    trades[trades < stop] = stop
    trades.cumsum().plot()
    trades = pd.DataFrame(trades, columns = ['net'])
    
    trades = trades[trades.abs() < 2]
    trades['pair'] = column
    
    
    
    print(column, 'mean =', trades['net'].mean())
    sums = sums.append(pd.DataFrame(trades))


plt.figure()
print(sums['net'].mean())
sums['net'] = sums['net'] - 0.00025
sums = sums.sort_index()
sums['net'].cumsum().plot()
ax.legend()

#%%
plot_df_net_plot = plot_df_net.copy()

plot_each = True
time_start = '2000-03'
if(plot_each == True):
    
    for pair in plot_df_net_plot:
        test = plot_df_net_plot[[pair]]/8+1.1
        #test['pair_roll'] = symbols_df[pair].rolling(roll).mean()
        test['pair'] = symbols_df[pair]
        test[time_start:].plot()
    
ax = plt.figure()
for pair in plot_df_net:
    test = sums[sums['pair'] == pair]
    test['net'][time_start:].cumsum().plot()

ax.legend()

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