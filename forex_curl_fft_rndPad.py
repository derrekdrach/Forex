# -*- coding: utf-8 -*-
"""
Created on Wed May 11 17:54:22 2022

@author: derre
"""


#%% User inputs
symbol_list = ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY', 'EURGBP']
symbol_list = [ 'EURUSD']

futures_list = ['ZR']
#symbol_list_den = ['EURUSD', 'AUDUSD']
#compare_symbol_list = ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY']

#Import standard modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.optimize import shgo

# import yahoo finance data
import yfinance as yf
# example: data_daily = yf.download(tickers = 'EURUSD=X', period  = '3y', interval = '1d')

# Import FRED data: Federal Reserve Economic Data
#import fredapi as fa
#example: fred = fa.Fred(api_key = '47aed364aa18b64e34f2f9695bd6dd67')
#%% commodity futures



#%%

symbols_df = pd.DataFrame()
symbols_df_high = pd.DataFrame()
symbols_df_low = pd.DataFrame()
period = '60d'
interval = '1h'


for symbol in symbol_list:
    print(symbol)

    df = yf.download(tickers = symbol + '=X', period  = period, interval = interval)
    
    #open_daily = yf.download(tickers = symbol + '=X', interval = interval, start = "2022-04-20", end = "2022-04-25")['Close']
    symbols_df[symbol] = df['Open']
    #symbols_df_high[symbol] = df['High']
    #symbols_df_low[symbol] = df['Low']
    #open_df = open_df/open_df.iloc[0]
    #open_df[open_df > 2] = float('NaN')
    

period = '60d'
interval = '5m'

for symbol in symbol_list:
    print(symbol)

    df = yf.download(tickers = symbol + '=X', period  = period, interval = interval)
    
    #open_daily = yf.download(tickers = symbol + '=X', interval = interval, start = "2022-04-20", end = "2022-04-25")['Close']
    
    symbols_df_high[symbol] = df['High']
    symbols_df_low[symbol] = df['Low']

   

symbols_df_price = symbols_df.copy()
symbols_df = symbols_df_price/symbols_df_price.iloc[0]
symbols_df_high = symbols_df_high/symbols_df_price.iloc[0]
symbols_df_low = symbols_df_low/symbols_df_price.iloc[0]

symbols_df[symbols_df > 2] = float('NaN')    
symbols_df_high[symbols_df_high > 2] = float('NaN')    
symbols_df_low[symbols_df_low > 2] = float('NaN') 

symbols_df = symbols_df.ffill()
symbols_df_high = symbols_df_high.ffill()
symbols_df_low = symbols_df_low.ffill()

symbols_df.index = pd.to_datetime(symbols_df.index)
symbols_df_high.index = pd.to_datetime(symbols_df_high.index)
symbols_df_low.index = pd.to_datetime(symbols_df_low.index)


symbols_df_dly = symbols_df.resample('d').first().ffill()

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
symbol_list_den = [ 'EURUSD']

symbols_df = pd.DataFrame()



file_directory = r'C:\Users\derre\Desktop\data\scripts'
for symbol in symbol_list_den:
    file_name = '_dataout.txt'
    open_df = pd.read_csv(file_directory + '\\' +  symbol + file_name, names = [symbol])
    file_name = '_timeStamp.txt'
    open_df.index = pd.read_csv(file_directory + '\\' + symbol + file_name, header = None).values.flatten()
    symbols_df[symbol] = open_df[symbol]
    
symbols_df = symbols_df/symbols_df.iloc[0]
symbols_df.index = pd.to_datetime(symbols_df.index)
symbols_df_dly = symbols_df.resample('d').first().dropna()
    
#%%
symbols_df_1m = pd.DataFrame()

for symbol in symbol_list_den:
    file_name = '_1m_dataout.txt'
    open_df = pd.read_csv(file_directory + '\\' +  symbol + file_name, names = [symbol])
    file_name = '_1m_timeStamp.txt'
    open_df.index = pd.read_csv(file_directory + '\\' + symbol + file_name, header = None).values.flatten()
    symbols_df_1m[symbol] = open_df[symbol]
    
symbols_df_1m = symbols_df_1m/symbols_df_1m.iloc[0]
symbols_df_1m.index = pd.to_datetime(symbols_df_1m.index)
symbols_df = symbols_df_1m.resample('h').first().dropna()

symbols_df_dly = symbols_df_1m.resample('d').first().dropna()


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

"""

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

# for curl method
x = 0.005
diff_thresh_dict = {'EURUSD':x,
                    'GBPUSD':x,
                    'AUDUSD':0.005,
                    'NZDUSD':0.025,
                    'USDCAD':x,
                    'USDCHF':x,
                    'USDJPY':x,
                    'EURGBP':x
                    }


x = 5
diff_range_dict = { 'EURUSD':x*25,
                    'GBPUSD':x*20,
                    'AUDUSD':x*20,
                    'NZDUSD':x*40,
                    'USDCAD':x*1,
                    'USDCHF':x*25,
                    'USDJPY':x*25,
                    'EURGBP':x*25}
#x = 5*26
diff_range_2_dict ={'EURUSD':x*25,
                    'GBPUSD':x*20,
                    'AUDUSD':x*40,
                    'NZDUSD':x*1,
                    'USDCAD':x*25,
                    'USDCHF':x*25,
                    'USDJPY':x*25,
                    'EURGBP':x*25
                    }



#direction = 1

plot_df_net_loop = symbols_df_dly.copy()
symbols_loop_df = symbols_df_dly.copy()
for symbol in symbols_loop_df:
    plot_df_net_loop[symbol] = np.sign(symbols_loop_df[symbol].rolling(roll).mean().diff(diff_range_dict[symbol]).diff(diff_range_2_dict[symbol]).shift(shift))
    plot_df_net_loop[symbol][symbols_loop_df[symbol].rolling(roll).mean().diff(diff_range_dict[symbol]).diff(diff_range_2_dict[symbol]).shift(shift).abs() < diff_thresh_dict[symbol]] = float('NaN')
    

plot_df_net_loop = plot_df_net_loop.ffill()    
plot_df_net_dly = plot_df_net_loop.copy()


method = 1

if(method == 1):
# for curl method - hourly
    roll = 1
    diff_thresh_dict = {'EURUSD':0.02,
                        'GBPUSD':0.02,
                        'AUDUSD':0.02,
                        'NZDUSD':0.02,
                        'USDCAD':0.02,
                        'USDCHF':0.02,
                        'USDJPY':0.02,
                        'EURGBP':0.02,
                        'EURGBP':0.02}
    
    
    standard_1 = 24*45
    standard_2 = 24*25
    standard_3 = 24*20
    standard_4 = 24*55
    standard_5 = 24*60
    standard_6 = 24*35
    standard_temp = standard_5
    
    
    diff_range_dict = { 'EURUSD':standard_5,
                        'GBPUSD':standard_3,
                        'AUDUSD':standard_4,
                        'NZDUSD':standard_3,
                        'USDCAD':standard_5,
                        'USDCHF':standard_temp,
                        'USDJPY':standard_1,
                        'EURGBP':standard_1
                        }

    x = 1
    diff_range_2_dict = diff_range_dict
                        #{'EURUSD':standard_1,
                        #'GBPUSD':standard_3,
                        #'AUDUSD':standard_4,
                        #'NZDUSD':standard_3,
                        #'USDCAD':standard_5,
                        #'USDCHF':standard_temp,
                        #'USDJPY':standard_1,
                        #'EURGBP':standard_1
                        #}
    standard_1 = 5e-3
    fft_filt_dict = {'EURUSD':standard_1,
                    'GBPUSD':standard_1,
                    'AUDUSD':standard_1,
                    'NZDUSD':standard_1,
                    'USDCAD':standard_1,
                    'USDCHF':standard_1,
                    'USDJPY':standard_1,
                    'EURGBP':standard_1
                    }
    direction = 1

elif(method == 2):
    # for diff method - hourly
    diff_thresh_dict = {'EURUSD':0.005,
                        'GBPUSD':0.0008,
                        'AUDUSD':0.00065,
                        'NZDUSD':0.00065,
                        'USDCAD':0.0007,
                        'USDCHF':0.0007,
                        'USDJPY':0.00065,
                        'EURGBP':0.00065}
    
    diff_range_dict =  {'EURUSD':1,
                        'GBPUSD':1,
                        'AUDUSD':1,
                        'NZDUSD':1,
                        'USDCAD':1,
                        'USDCHF':1,
                        'USDJPY':1,
                        'EURGBP':1}
    
    roll_dict =        {'EURUSD':12,
                        'GBPUSD':24*10,
                        'AUDUSD':24*4,
                        'NZDUSD':24*6,
                        'USDCAD':24*5,
                        'USDCHF':24*15,
                        'USDJPY':24*15,
                        'EURGBP':24*10}
    
    roll_dict =        {'EURUSD':12,
                        'GBPUSD':12,
                        'AUDUSD':18,
                        'NZDUSD':18,
                        'USDCAD':24,
                        'USDCHF':18,
                        'USDJPY':18,
                        'EURGBP':18}
    
    

    direction = 1


plot_df_net_loop = symbols_df.copy()
symbols_loop_df = symbols_df.copy()

if(method < 2):
    # curl
    print('curl')
    for symbol in symbols_df:
        plot_df_net_loop[symbol] = (symbols_loop_df[symbol].rolling(roll).mean().diff(diff_range_dict[symbol]).diff(diff_range_2_dict[symbol])).rolling(1).mean().diff()
        fft = np.fft.fft(plot_df_net_loop[symbol].fillna(0).values)
        filt = fft_filt_dict[symbol]
        filt = int(filt*len(fft))
        fft_filtered = fft
        fft_filtered[filt:-filt] = 0
        curl_data_filt = (np.fft.ifft(fft_filtered))
        curl_data_filt_df = pd.Series(curl_data_filt, index = plot_df_net_loop.index)
        plot_df_net_loop[symbol] = np.real(curl_data_filt_df)
        plot_df_net_loop = np.sign((plot_df_net_loop))
        #plot_df_net_loop[symbol][symbols_loop_df[symbol].rolling(roll).mean().diff(diff_range_dict[symbol]).diff(diff_range_2_dict[symbol]).shift(shift).rolling(standard_1).mean().diff().abs() <  diff_thresh_dict[symbol]] = float('NaN')
    
else:
    # diff
    for symbol in symbols_df:
        plot_df_net_loop[symbol] = np.sign(symbols_loop_df[symbol].diff(diff_range_dict[symbol]).rolling(roll_dict[symbol]).mean())
        plot_df_net_loop[symbol][symbols_loop_df[symbol].diff(diff_range_dict[symbol]).rolling(roll_dict[symbol]).mean().abs() < diff_thresh_dict[symbol]] = float('NaN')






# break to inject data
#%%

plot_df_net_loop = pd.DataFrame()
plot_df_net_loop['EURUSD'] = plot_curl_pair_df['tail'].copy() #
#plot_df_net_loop = plot_df_net_loop.ffill().shift(24*2)





plot_df_net = plot_df_net_loop.copy()
plot_df_net = plot_df_net*direction


plot_df_net_dly_on_hr = plot_df_net_dly.resample('H').ffill()
#plot_df_net_sum = plot_df_net_dly_on_hr.add(plot_df_net)
plot_df_net_events = np.sign(plot_df_net_dly_on_hr.add(plot_df_net).abs().ffill().diff())
plot_df_net_events = plot_df_net_events.loc[plot_df_net.index]
plot_df_net_events = plot_df_net_events[~plot_df_net_events.index.duplicated(keep = 'first')]
#plot_df_net_events = np.sign(plot_df_net.diff())

"""
symbol = 'GBPUSD'
plot_df_net_events[symbol][plot_df_net_events[symbol].abs() >0].iloc[::2].diff().plot()
test = pd.DataFrame(index = plot_df_net_events.index)
test['drop'] = plot_df_net_events[symbol][plot_df_net_events[symbol].abs() >0].iloc[::2].diff()
plot_df_net_events[symbol][test['drop'].abs() > 0] = 0
plot_df_net_events[symbol][plot_df_net_events[symbol].abs() >0].iloc[::2].diff().plot()
"""


#drop_index = plot_df_net_events[plot_df_net_events.cumsum() == 2]['EURUSD'].dropna().index[0]
#plot_df_net_events[plot_df_net_events.index == drop_index] = float('NaN')



plot_df_net_entries = pd.DataFrame()
plot_df_net_exits = pd.DataFrame()
for symbol in symbols_df:    
    if(False):
        drops = pd.DataFrame(index = plot_df_net_events.index)
        drops['drop_0'] = plot_df_net_events[symbol][plot_df_net_events[symbol].abs() > 0].iloc[0::2].diff()
        drops['drop_1'] = plot_df_net_events[symbol][plot_df_net_events[symbol].abs() > 0].iloc[1::2].diff()
        
        plot_df_net_events[symbol][drops['drop_0'].abs() > 0] = 0
        plot_df_net_events[symbol][drops['drop_1'].abs() > 0] = 0
        #plot_df_net_events = plot_df_net_events.loc[plot_df_net.index]
        
    plot_df_net_entries = pd.concat([plot_df_net_entries, plot_df_net[symbol][plot_df_net_events[symbol] > 0]], axis = 1)
    plot_df_net_exits = pd.concat([plot_df_net_exits, plot_df_net[symbol][plot_df_net_events[symbol] < 0]], axis = 1)
    
        


stop = -0.15 #see flag
limit = 0.15 #see flag
stop_limit_flag = False
sums = pd.DataFrame()

for symbol in symbols_df:
        
    
    symbol_entries = plot_df_net_entries[symbol].dropna()
    symbol_entries = -1*symbol_entries*symbols_df[symbol].loc[symbol_entries.index]
    
    
    symbol_exits = plot_df_net_exits[symbol].dropna()
    symbol_exits = symbol_exits[symbol_exits.index > symbol_entries.index[0]]
    if(symbol_entries.index[-1] > symbol_exits.index[-1]):
        print('clip')
        symbol_exits = pd.concat([symbol_exits, (symbols_df.iloc[[-1]][symbol]*0+1)])
    symbol_exits = symbol_exits*symbols_df[symbol].loc[symbol_exits.index]
    
    
    
        
    
    
    
    symbol_net = pd.DataFrame(symbol_entries.add((symbol_exits.abs()*-1*np.sign(symbol_entries).values).values))
    symbol_net = symbol_net.rename(columns = {symbol:'net'})
    symbol_net = symbol_net.rename(columns = {0:'net'})

    
    if(stop_limit_flag):
        for index, item in enumerate(symbol_entries):   
            
            stop_flag = False
            limit_flag = False
            
            
            
            
            if(symbol_entries[index] < 0):
                diff_open = symbols_df_1m[symbol].loc[symbol_entries.index[index]:symbol_exits.index[index]].min() + symbol_entries[index]
                stop_time = symbols_df_1m[symbol].loc[symbol_entries.index[index]:symbol_exits.index[index]].idxmin()
                if(np.abs(diff_open) > np.abs(stop)):
                    #print('stop: ', diff_open)
                    symbol_net['net'].iloc[index] = stop
                    stop_flag = True
                    
            if(symbol_entries[index] > 0):
                diff_open = symbols_df_1m[symbol].loc[symbol_entries.index[index]:symbol_exits.index[index]].max() - symbol_entries[index]
                stop_time = symbols_df_1m[symbol].loc[symbol_entries.index[index]:symbol_exits.index[index]].idxmax()
                if(np.abs(diff_open) > np.abs(stop)):
                    #print('stop: ', diff_open)
                    symbol_net['net'].iloc[index] = stop
                    stop_flag = True
            
            #if(stop_flag == 0):  
            
            if(symbol_entries[index] < 0):
                diff_open = symbols_df_1m[symbol].loc[symbol_entries.index[index]:symbol_exits.index[index]].max() + symbol_entries[index]
                limit_time = symbols_df_1m[symbol].loc[symbol_entries.index[index]:symbol_exits.index[index]].idxmax()
                if(np.abs(diff_open) > np.abs(limit)):
                    #print('Short limit: ', diff_open)
                    symbol_net['net'].iloc[index] = limit
                    limit_flag = True
            
            
            if(symbol_entries[index] > 0):
                diff_open = symbols_df_1m[symbol].loc[symbol_entries.index[index]:symbol_exits.index[index]].min() - symbol_entries[index]
                limit_time = symbols_df_1m[symbol].loc[symbol_entries.index[index]:symbol_exits.index[index]].idxmin()
                if(np.abs(diff_open) > np.abs(limit)):
                    #print('Long limit: ', diff_open, limit)
                    symbol_net['net'].iloc[index] = limit
                    limit_flag = True
                        
            if(stop_flag == True & limit_flag ==True):
                
                if(stop_time < limit_time):
                    symbol_net['net'].iloc[index] = stop
                elif(stop_time > limit_time):
                    symbol_net['net'].iloc[index] = limit
                else:
                    print('conflict')
                    
                    

                
                
    #symbol_net[symbol_net < stop] = stop
    print(symbol, "sum", (symbol_net['net']-.00025).sum())
    symbol_net['pair'] = symbol
    sums = sums.append(symbol_net)
    

#sums[sums['net'] < stop]['net'] = stop
sums = sums.sort_index()
sums['net'] = sums['net'] - 0.00025
sums['net'].cumsum().plot()
print('sum = ', sums['net'].sum())
print('mean = ', sums['net'].mean())








plot_df_net_plot = plot_df_net[plot_df_net_dly_on_hr.add(plot_df_net).abs() > 0].fillna(0)
#plot_df_net_plot = plot_df_net_dly_on_hr.copy()

plot_each = False
time_start = '2000'
if(plot_each == True):
    
    for pair in plot_df_net_plot:
        ax = plt.figure()
        test = plot_df_net_plot[[pair]]/8+1.1
        #test['pair_roll'] = symbols_df[pair].rolling(roll).mean()
        test['pair'] = symbols_df[pair]
        test[time_start:].plot()
        #ax.legend()
    
ax = plt.figure()
for pair in plot_df_net:
    test = sums[sums['pair'] == pair]
    test['net'][time_start:].cumsum().plot()

ax.legend()


#%% extra visualizations



symbol = 'EURUSD'
curl_data_init = (symbols_loop_df[symbol].rolling(roll).mean().diff(diff_range_dict[symbol]).diff(diff_range_2_dict[symbol]).shift(shift))
curl_fft_slope = pd.DataFrame(columns = np.arange(-48,1,6))#np.zeros(len(curl_data_init))
#curl_fft_slope = np.zeros(len(curl_data_init))
curl_fft_slope_merit = np.zeros(len(curl_data_init))
end_init = '2022-01-01'
end_final = '2022-05-20'

region_of_interest = curl_data_init.loc[end_init:end_final:6].index
filt_val = 20e-3
for idx_1, end_datetime in enumerate(region_of_interest):
    start_offset_list = np.arange(-1,-360*24,-1)
    start_offset_list = random.sample(start_offset_list.tolist(), 359*24)
    start_offset_merit = 100
    best_offset = 0
    
    #start_offset = [-24*180]
    #p_bounds = [(-24*180, 24*180)]
    #test = shgo(fun, args = (end_datetime, filt_val, curl_data_init), bounds = p_bounds, options = { 'maxiter': 1000})
    #print(test.x, test.fun)


    for idx_2, each in enumerate(start_offset_list):
        
        time_end = pd.to_datetime(end_datetime) + pd.DateOffset(days = 0)
        time_start = time_end - pd.DateOffset(years = 2) + pd.DateOffset(hours = int(0))
        time_end_pad = pd.to_datetime(end_datetime) + pd.DateOffset(hours = int(each))
        time_start_pad = time_end_pad + pd.DateOffset(days = -14)
        
        curl_data = curl_data_init[time_start:time_end].copy()
        pad_data = curl_data_init[time_start_pad:time_end_pad].copy()
        pad_data = pad_data - pad_data[0] + curl_data[-1]
        curl_data_pad = pd.concat([curl_data, pad_data])
        
        fft = np.fft.fft(curl_data_pad.dropna().values)
        filt = int(filt_val*len(fft))
        fft_filtered = fft
        fft_filtered[filt:-filt] = 0
        curl_data_filt = np.fft.ifft(fft_filtered)
        curl_data_filt_df = (pd.Series(np.real(curl_data_filt), index = curl_data_pad.dropna().index))
        #curl_data_filt_df.plot()
        curl_data_filt_df = curl_data_filt_df.sort_index().loc[time_start:time_end]
        
        
    
        
        offset_merit_new = np.abs((curl_data_filt_df[-24*20:] - curl_data[-24*20:]).sum())
        
        if(offset_merit_new < start_offset_merit):
            start_offset_merit = offset_merit_new
            best_offset = start_offset_list[idx_2]
    
        if(offset_merit_new < 1e-1):
            break
        #T0 = 320
        #f0 = 1/T0
        #length = len(curl_data)
        #x_vals = np.arange(0, length, 1)
        #offset = 100
    #plot_curl_pair_df = pd.DataFrame()
    #plot_curl_pair_df['curl_fft'] = curl_data_filt_df/curl_data_filt_df.max()
    #plot_curl_pair_df['curl'] = curl_data/curl_data.max()
    #plot_curl_pair_df['sign'] = np.sign(curl_data_filt_df.diff())
    #plot_curl_pair_df['pair'] = symbols_df[symbol]
    curl_fft_slope.loc[end_datetime] = ((curl_data_filt_df/curl_data_filt_df.abs().max()).diff().iloc[np.arange(-49,-0,6)].values)

    #curl_fft_slope[idx_1] = (curl_data_filt_df/curl_data_filt_df.abs().max()).diff()[-12]
    curl_fft_slope_merit[idx_1] = offset_merit_new
    print(idx_1, "/", len(region_of_interest), ":   ", offset_merit_new, ":   ", curl_fft_slope[-36][end_datetime])

plot_curl_pair_df = pd.DataFrame()
plot_curl_pair_df['curl_fft'] = curl_data_filt_df/curl_data_filt_df.abs().max()
plot_curl_pair_df['curl'] = curl_data/curl_data.max()
plot_curl_pair_df['sign'] = np.sign(curl_data_filt_df.diff())
plot_curl_pair_df['pair'] = symbols_df[symbol]
        #wave_plot = np.sin(f0*x_vals+ offset)
        
        
        #plot_curl_pair_df['wave'] = wave_plot
        
    #print("best =", best_offset)

#%%
np.sign(curl_fft_slope[[-12]]).plot()
#(curl_fft_slope).plot()

#curl_fft_slope.mean(axis = 1).plot()
#symbols_df[symbol].loc[curl_fft_slope.index].plot(secondary_y = True)
(curl_data_filt_df/curl_data_filt_df.abs().max()).loc['2022-01-01':'2022-05-20'].plot(secondary_y = True)

#%% re-format to view a static timestamp from multiple future hours.
curl_fft_slope_shift = curl_fft_slope[[-48, -42, -36, -30, -24, -18, -12]].copy()
for idx_1, item in curl_fft_slope_shift.iteritems():
    curl_fft_slope_shift[idx_1] = curl_fft_slope_shift[idx_1].shift(int(idx_1/6))
curl_fft_slope_shift.mean(axis = 1).plot()
#curl_fft_slope_shift.plot()
#%%  
for idx_1, item in curl_fft_slope_shift.iterrows():
    curl_fft_slope_shift.loc[idx_1].plot()
    
#%%
def fun(start_offset, end_datetime, filt_val, curl_data_init):
    time_end = pd.to_datetime(end_datetime) + pd.DateOffset(hours = 0)
    time_start = time_end - pd.DateOffset(years = 2) + pd.DateOffset(hours = int(start_offset))
    curl_data = curl_data_init[time_start:time_end].copy()
    
    fft = np.fft.fft(curl_data.dropna().values)
    filt = int(filt_val*len(fft))
    fft_filtered = fft
    fft_filtered[filt:-filt] = 0
    curl_data_filt = np.fft.ifft(fft_filtered)
    curl_data_filt_df = (pd.Series(np.real(curl_data_filt), index = curl_data.dropna().index))
    #curl_data_filt_df.plot()
    offset_merit_new = np.abs((curl_data_filt_df[-24*40:] - curl_data[-24*40:]).sum())
    return offset_merit_new

def fun_new(date_time_list, start_offset):
    
#%%
fun_return = fun(0, end_datetime, filt_val, curl_data_init)

#%% Make a copy
curl_fft_slope_copy = curl_fft_slope.copy()

#%%
curl_fft_slope = curl_fft_slope_copy.copy()
#%%

    
#end_init = '2022-02-01'
#end_final = '2022-03-01'
#region_of_interest = curl_data_init.loc[end_init:end_final:1].index

curl_fft_slope = curl_fft_slope_copy[-48].copy()
curl_fft_slope_series = pd.Series(curl_fft_slope).copy()
curl_fft_slope_series = curl_fft_slope_series[curl_fft_slope_series.abs()>0].rolling(1).mean()
curl_fft_slope_series.index = region_of_interest[:]
curl_fft_slope_series[curl_fft_slope_series.abs() < 0.00001 ] = float('Nan')


plot_curl_pair_df['tail'] = curl_fft_slope_series.copy()
plot_curl_pair_df['tail'] = plot_curl_pair_df['tail'].ffill()
plot_curl_pair_df['tail'] = np.sign(plot_curl_pair_df['tail'])
plot_curl_pair_df['tail'] = np.sign((plot_curl_pair_df['tail'].ffill().rolling(1).mean().shift(-0)))


#curl_fft_merit_series = pd.Series(curl_fft_slope_merit)
#curl_fft_merit_series = curl_fft_merit_series[curl_fft_merit_series.abs()>0]
#curl_fft_merit_series.index = region_of_interest[:]
#plot_curl_pair_df['merit'] = curl_fft_merit_series/curl_fft_merit_series.max()
#plot_curl_pair_df['merit'] = plot_curl_pair_df['merit'].ffill()


plot_curl_pair_df['tail'].plot(secondary_y = True)
#plot_curl_pair_df['merit'].plot(secondary_y = True)

plot_curl_pair_df['curl_fft'][end_init:].shift(0).plot(secondary_y = True)
plot_curl_pair_df['pair'] = symbols_df[symbol].loc[end_init:]
plot_curl_pair_df['pair'][end_init:].plot()
#(plot_curl_pair_df['curl_fft'][:].shift(24*95)).plot(secondary_y = True)
plot_curl_pair_df['curl'][end_init:].rolling(1).mean().shift(0).plot(secondary_y = True)
#plot_curl_pair_df['wave'][:].rolling(1).mean().shift(24*20).plot(secondary_y = True)

#plot_curl_pair_df['curl'][:].rolling(24*0).mean().shift(0).plot(secondary_y = True)

#plot_curl_pair_df['sign'] = plot_curl_pair_df['curl'].rolling(3).mean().diff().copy()
#plot_curl_pair_df['sign'][plot_curl_pair_df['sign'].abs() < 0.02] = float('Nan')
#plot_curl_pair_df['sign'] = plot_curl_pair_df['sign'].ffill()
#plot_curl_pair_df['sign'] = np.sign(plot_curl_pair_df['sign'])
#plot_curl_pair_df['sign'][end_init:].shift(0).plot(secondary_y = True)



#plot_curl_pair_df['sign'].plot(secondary_y = True)

 


#%% find time delta between peaks and plot histogram
symbol = 'EURUSD'

plot_df_net_loop_test = (symbols_loop_df[symbol].rolling(roll).mean().diff(diff_range_dict[symbol]).diff(diff_range_2_dict[symbol])).rolling(1).mean()
fft = np.fft.fft(plot_df_net_loop_test.fillna(0).values)
filt = int(10e-3*len(fft))
fft_filtered = fft
fft_filtered[filt:-filt] = 0
curl_data_filt = ((np.fft.ifft(fft_filtered)))
curl_data_filt_df = pd.Series((curl_data_filt), index = plot_df_net_loop_test.index)
#curl_data_filt_df.plot()

plot_df_net_loop_peaks = np.sign(curl_data_filt_df.diff())
plot_df_net_loop_peaks = pd.Series(np.real(plot_df_net_loop_peaks), index = plot_df_net_loop_peaks.index)
plot_df_net_loop_peaks = plot_df_net_loop_peaks.diff()
#plot_df_net_loop_peaks.plot()

plot_df_net_loop_peaks = plot_df_net_loop_peaks[plot_df_net_loop_peaks.abs() > 0]
plot_df_net_loop_peaks = pd.Series(plot_df_net_loop_peaks.index).diff()

for index, peaks in plot_df_net_loop_peaks.iteritems():
    plot_df_net_loop_peaks.iloc[index] = peaks.days
    #print(peaks.days)

plot_df_net_loop_peaks = plot_df_net_loop_peaks[1:]



plot_df_net_loop_peaks.plot.hist(bins = 40)
#plot_df_net_loop_peaks.rolling(1).mean().plot()
#plot_df_net_loop_peaks.sort_values().reset_index(drop = True).cumsum().plot()