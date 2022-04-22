# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 19:16:32 2022

@author: derre
"""


#%% user inputs:
year = '1999'

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

# Import forex data from yfinance
data = yf.download(tickers = 'EURUSD=X', period  = '25y', interval = '1mo')
data_SP500_usd = yf.download(tickers = '^GSPC', period  = '25y', interval = '1mo')
data_FTSE_eur = yf.download(tickers = '^FTSE', period  = '25y', interval = '1mo')



#%% generate monthly period dataframes
data_M = data.copy()
data_M.index = data_M.index.to_period('M')
data_M = data_M.groupby(data_M.index).first()

data_SP500_usd_M = data_SP500_usd.copy()
data_SP500_usd_M.index = data_SP500_usd_M.index.to_period('M')
data_SP500_usd_M = data_SP500_usd_M.groupby(data_SP500_usd_M.index).first()
data_SP500_usd_M = pd.DataFrame(data_SP500_usd_M['Open']).rename(columns = {'Open': 'SP500_usd'})

data_FTSE_eur_M = data_FTSE_eur.copy()
data_FTSE_eur_M.index = data_FTSE_eur_M.index.to_period('M')
data_FTSE_eur_M = data_FTSE_eur_M.groupby(data_FTSE_eur_M.index).first()
data_FTSE_eur_M = pd.DataFrame(data_FTSE_eur_M['Open']).rename(columns = {'Open': 'FTSE_eur'})



#%% download fed data from FRED
#gdp_usd = fred.get_series('GDPC1') #real gdp
gdp_usd = fred.get_series('GDP')*1e3 #convert B to M
#gdp_eur = fred.get_series('CLVMNACSCAB1GQEU28') #real gdp
gdp_eur = fred.get_series('CPMNACSCAB1GQEU272020')
m1_usd = fred.get_series('M1SL')
m2_usd = fred.get_series('M2SL')
m3_usd = fred.get_series('MABMM301USM189S')
m1_eur = fred.get_series('MANMM101EZM189N')
m2_eur = fred.get_series('MYAGM2EZM196N')/1e9
m3_eur = fred.get_series('MABMM301EZM189S')#/1e9
assets_usd = fred.get_series('WALCL')
assets_eur = fred.get_series('ECBASSETSW')
rate_eur = fred.get_series('ECBDFR')
rate_usd = fred.get_series('FEDFUNDS')
cpi_usd = fred.get_series('CPALTT01USM659N')
cpi_eur = fred.get_series('EA19CPALTT01GYM')
debt_per_gdp_usd = fred.get_series('GFDEGDQ188S')
file_path = r'C:\Users\derre\Downloads'
file_name = '\\amCharts.csv'


#%%
debt_per_gdp_eur = pd.read_csv(file_path + file_name)
debt_per_gdp_eur['date'] = debt_per_gdp_eur['date'].str[:10]
debt_per_gdp_eur['date'] = pd.to_datetime(debt_per_gdp_eur['date'])
debt_per_gdp_eur.set_index('date', inplace = True)
debt_per_gdp_eur.index = debt_per_gdp_eur.index.to_period('M')
debt_per_gdp_eur = pd.DataFrame(debt_per_gdp_eur['s1']).rename(columns = {'s1':'debt_per_gdp_eur'})


debt_per_gdp_usd.index = debt_per_gdp_usd.index.to_period('M')
debt_per_gdp_usd = pd.DataFrame(debt_per_gdp_usd, columns = ['debt_per_gdp_usd'])#.rename(columns = {'0':'debt_per_gdp_usd'})

debt_per_gdp_ratio = pd.concat([debt_per_gdp_eur, debt_per_gdp_usd], axis = 1).ffill()
debt_per_gdp_ratio = pd.DataFrame(debt_per_gdp_ratio['debt_per_gdp_usd']/debt_per_gdp_ratio['debt_per_gdp_eur'], columns = ['debt_per_gdp_ratio'])

#%% generate monthly period dataframes

SP500_per_gdp = pd.concat([data_SP500_usd_M, gdp_usd_M], axis = 1).ffill()
SP500_per_gdp = SP500_per_gdp['SP500_usd']/SP500_per_gdp['gdp_usd']

FTSE_per_gdp = pd.concat([data_FTSE_eur_M, gdp_eur_M], axis = 1).ffill()
FTSE_per_gdp = FTSE_per_gdp['FTSE_eur']/FTSE_per_gdp['gdp_eur']

SP500_FTSE_ratio = pd.DataFrame(FTSE_per_gdp/SP500_per_gdp, columns = ['SP500_FTSE_ratio'])

#%%

cpi_usd_M = cpi_usd.copy()
cpi_usd_M.index = cpi_usd.index.to_period('M')

cpi_eur_M = cpi_eur.copy()
cpi_eur_M.index = cpi_eur.index.to_period('M')

cpi_ratio = pd.DataFrame(cpi_eur_M-cpi_usd_M, columns = ['cpi_ratio'])

gdp_usd_M = gdp_usd.copy()
gdp_usd_M.index = gdp_usd_M.index.to_period('M')
gdp_usd_M = pd.DataFrame(gdp_usd_M.groupby(gdp_usd_M.index).first(), columns = ['gdp_usd'])

gdp_eur_M = gdp_eur.copy()
gdp_eur_M.index = gdp_eur_M.index.to_period('M')
gdp_eur_M = pd.DataFrame(gdp_eur_M.groupby(gdp_eur_M.index).first(), columns = ['gdp_eur'])

gdp_ratio = gdp_eur/gdp_usd
gdp_ratio.index = gdp_ratio.index.to_period('M')
gdp_ratio = pd.DataFrame(gdp_ratio.groupby(gdp_ratio.index).first().dropna(), columns = ['gdp_ratio'])

rate_eur_M = rate_eur.copy()
rate_eur_M.index = rate_eur_M.index.to_period('M')
rate_eur_M = pd.DataFrame(rate_eur_M.groupby(rate_eur_M.index).first(), columns = ['rate_eur_M'])


rate_usd_M = rate_usd.copy()
rate_usd_M.index = rate_usd_M.index.to_period('M')
rate_usd_M = pd.DataFrame(rate_usd_M.groupby(rate_usd_M.index).first(), columns = ['rate_usd_M'])

# combine rates into single dataframe
rates = pd.concat([rate_eur_M, rate_usd_M], axis = 1)
#rates.index = rates.index.to_period('M')
#rates = rates.groupby(rates.index).first()

#find rate ratio
rate_ratio = pd.DataFrame(((rate_eur/rate_usd).dropna()/10).loc[year:], columns = ['rate_ratio'])
rate_ratio.index = rate_ratio.index.to_period('M')
rate_ratio = rate_ratio.groupby(rate_ratio.index).first()

rate_dif = pd.DataFrame(((rate_eur-rate_usd*1.25).dropna()).loc[year:], columns = ['rate_dif'])
rate_dif.index = rate_dif.index.to_period('M')
rate_dif = rate_dif.groupby(rate_dif.index).first()




#%% combine m3 datasets into single dataframe
plot_df = pd.concat([m3_usd, m3_eur], axis = 1)
plot_df[year:].plot()

#%% generate monthly period dataframes

assets_eur_slice = assets_eur.loc[year:]
assets_eur_slice.index = assets_eur_slice.index.to_period('M')
assets_eur_slice = assets_eur_slice.groupby(assets_eur_slice.index).first()

assets_usd_slice = assets_usd.loc[year:]
assets_usd_slice.index = assets_usd_slice.index.to_period('M')
assets_usd_slice = assets_usd_slice.groupby(assets_usd_slice.index).first()

assets_per_gdp_usd = (assets_usd_slice/gdp_usd_M['gdp_usd']).dropna()
assets_per_gdp_eur = (assets_eur_slice/gdp_eur_M['gdp_eur']).dropna()
assets_per_gdp_ratio = pd.DataFrame(assets_per_gdp_eur/assets_per_gdp_usd, columns = ['assets_per_gdp_ratio'])
assets_per_gdp_diff = pd.DataFrame(assets_per_gdp_usd-assets_per_gdp_eur, columns = ['assets_per_gdp_diff'])

#%%
# find asset ratio
asset_ratio = pd.DataFrame((assets_eur_slice/assets_usd_slice), columns = ['asset_ratio'])

#slice currency pair data
pair_data = data_M['Open'].loc[year:]



#slice m1 data and convert to monthly period dataframe
m1_eur_slice = m1_eur.loc[year:]
m1_eur_slice.index = m1_eur_slice.index.to_period('M')

m1_usd_slice = m1_usd.loc[year:]
m1_usd_slice.index = m1_usd_slice.index.to_period('M')


#slice m3 data and convert to monthly period dataframe
m3_eur_slice = m3_eur.loc[year:]
m3_eur_slice.index = m3_eur_slice.index.to_period('M')

m3_usd_slice = m3_usd.loc[year:]
m3_usd_slice.index = m3_usd_slice.index.to_period('M')

#find m3 ratio
m1_ratio = pd.DataFrame(m1_eur_slice/m1_usd_slice, columns = ['m1_ratio'])
m3_ratio = pd.DataFrame(m3_eur_slice/m3_usd_slice, columns = ['m3_ratio'])




#%%
test_in = rate_dif/20
test = pd.concat([ gdp_ratio/gdp_ratio.max(), m3_ratio], axis = 1).ffill()
test = pd.DataFrame(test.mean(axis = 1), columns = ['test'])

plot_df = pd.concat([debt_per_gdp_ratio, SP500_FTSE_ratio, assets_per_gdp_diff, asset_ratio, m3_ratio, pair_data, rate_dif/rate_dif.max(), rate_usd_M, rate_eur_M, gdp_ratio/gdp_ratio.max(), test, m1_ratio, cpi_ratio*-1], axis = 1).ffill()
ax = plt.figure()
#plot_df['asset_ratio'].plot(legend = True)
(plot_df['m3_ratio']+plot_df['assets_per_gdp_diff']*0.3).plot()
#plot_df['m1_ratio'].plot()
plot_df['Open'].plot(secondary_y = True)
#(plot_df['rate_dif'].loc[year:]*0.25).plot()
#plot_df['gdp_ratio'].plot()
#(plot_df['m3_ratio']/plot_df['gdp_ratio']).plot()
#plot_df['test'].plot()
#(plot_df['rate_usd_M'].loc[year:]/10+1).plot()
#(plot_df['rate_eur_M'].loc[year:]/10+1).plot()
((plot_df['debt_per_gdp_ratio']*-1*0 + plot_df['assets_per_gdp_diff']*0.3 + plot_df['rate_dif'].loc[year:]*0.25 + plot_df['cpi_ratio']*0.15+plot_df['m3_ratio'])).plot()
(0.15*plot_df['cpi_ratio']+1).plot()
#((assets_per_gdp_ratio).diff()*100+1).plot()
ax.legend()

#%%
ax2 = plt.figure()
cpi_usd_M.plot()
cpi_eur_M.plot()

#%%
#(assets_per_gdp_eur).plot()
#(assets_per_gdp_usd).plot()
#(-assets_per_gdp_eur-assets_per_gdp_usd).plot()
(((assets_per_gdp_usd-assets_per_gdp_eur))*0.3+0.75).plot()