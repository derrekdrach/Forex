# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 19:16:32 2022

@author: derre
"""

#%%Import standard modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import plotly.graph_objs as go

# Import FRED data: Federal Reserve Economic Data
import fredapi as fa
#from local_settings import fred as settings
fred = fa.Fred(api_key = '47aed364aa18b64e34f2f9695bd6dd67')


#%% Functions

# Import forex data 

data = yf.download(tickers = 'EURUSD=X', period  = '15y', interval = '1d')

#%%
data['Open'].plot()
data_M = data.copy()
data_M.index = data_M.index.to_period('M')
data_M = data_M.groupby(data_M.index).first()



#%% download fed data
gdp = fred.get_series('GDP')
m2_usd = fred.get_series('M2SL')
m3_usd = fred.get_series('MABMM301USM189S')
m2_eur = fred.get_series('MYAGM2EZM196N')/1e9
m3_eur = fred.get_series('MABMM301EZM189S')#/1e9
assets_usd = fred.get_series('WALCL')
assets_eur = fred.get_series('ECBASSETSW')
rate_eur = fred.get_series('ECBDFR')
rate_usd = fred.get_series('FEDFUNDS')

#%%
rate_eur_M = rate_eur.copy()
rate_eur_M.index = rate_eur_M.index.to_period('M')
rate_eur_M = rate_eur_M.groupby(rate_eur_M.index).first()

rate_usd_M = rate_usd.copy()
rate_usd_M.index = rate_usd_M.index.to_period('M')
rate_usd_M = rate_usd_M.groupby(rate_usd_M.index).first()


rates = pd.concat([rate_eur_M, rate_usd_M], axis = 1)
#rates.index = rates.index.to_period('M')
#rates = rates.groupby(rates.index).first()
rate_ratio = (rate_eur/rate_usd).dropna()/10


#%%
plot_df = pd.concat([m3_usd, m3_eur], axis = 1)
#plot_df = plot_df.loc['2016':]
#plot_df = plot_df.groupby(by = [plot_df.index.month, plot_df.index.year]).sum()
plot_df['2016':].plot()

#%%
year = '2008'
assets_eur_slice = assets_eur.loc[year:]
assets_eur_slice.index = assets_eur_slice.index.to_period('M')
assets_eur_slice = assets_eur_slice.groupby(assets_eur_slice.index).first()

assets_usd_slice = assets_usd.loc[year:]
assets_usd_slice.index = assets_usd_slice.index.to_period('M')
assets_usd_slice = assets_usd_slice.groupby(assets_usd_slice.index).first()

#%%

asset_ratio = (assets_eur_slice/assets_usd_slice)
#divide_df.plot()
pair_data = data_M['Open'].loc[year:]

m3_eur_slice = m3_eur.loc[year:]
m3_eur_slice.index = m3_eur_slice.index.to_period('M')

m3_usd_slice = m3_usd.loc[year:]
m3_usd_slice.index = m3_usd_slice.index.to_period('M')

m3_ratio = m3_eur_slice/m3_usd_slice


#%%

plot_df = pd.concat([asset_ratio, m3_ratio, pair_data], axis = 1)
plt.figure()
plot_df[0].plot()
plot_df[1].plot()
plot_df['Open'].plot(secondary_y = True)
rate_ratio.plot()
