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
data = yf.download(tickers = 'AUDUSD=X', period  = '25y', interval = '1mo')
data_daily = yf.download(tickers = 'EURUSD=X', period  = '3y', interval = '1d')
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

data_FTSE_SP500_M_diff = pd.concat([data_SP500_usd_M, data_FTSE_eur_M], axis = 1)
data_FTSE_SP500_M_diff = data_FTSE_SP500_M_diff['FTSE_eur']-data_FTSE_SP500_M_diff['SP500_usd']
data_FTSE_SP500_M_diff = pd.DataFrame(data_FTSE_SP500_M_diff/data_FTSE_SP500_M_diff.max(), columns = ['data_FTSE_SP500_M_diff'])

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

PCEPI = fred.get_series('PCEPI')
PCEPI_core = fred.get_series('CORESTICKM159SFRBATL')
PCE_usd = fred.get_series('PCE')
PCE_usd_rolling = pd.DataFrame(PCE_usd.diff(periods = 12)/PCE_usd, columns = ['PCE_usd'])

PCE_eur = fred.get_series('NAEXKP02EZQ189S')
PCE_eur_rolling = pd.DataFrame(PCE_eur.diff(periods = 12)/PCE_eur, columns = ['PCE_eur'])
PCE_diff = pd.concat([PCE_usd_rolling, PCE_eur_rolling],axis = 1).interpolate()
PCE_diff = PCE_diff['PCE_usd']-PCE_diff['PCE_eur']
PCE_diff.index = PCE_diff.index.to_period('M')
PCE_diff = pd.DataFrame(PCE_diff, columns = ['PCE_diff'])

CPI_Energy_usd = fred.get_series('CPIENGSL')
CPI_Energy_eur = fred.get_series('ELGAS0EUCCM086NEST')
CPI_Energy_eur_2 = fred.get_series('ENRGY0EZ19M086NEST')

CPI_Core_usd = fred.get_series('CPILFESL')
#CPI_Core_eur

exports_usd_quarterly = fred.get_series('EXPGS') # BOPTEXP monthly
exports_usd = fred.get_series('BOPTEXP') # BOPTEXP monthly

exports_eur_quarterly = fred.get_series('XTEXVA01EZQ664S')#'XTEXVA01EZM667S') monthly 
exports_eur = fred.get_series('XTEXVA01EZM667S')#'XTEXVA01EZM667S') monthly
insert_time = pd.to_datetime('2022-02-01')

exports_eur.loc[insert_time] = 2.537e11

#%%
exports_ratio = pd.DataFrame()
exports_ratio['exports_ratio'] = (((exports_usd/exports_usd.max())/(exports_eur/exports_eur.max()))**-1).dropna()
exports_ratio.index = exports_ratio.index.to_period('M')

#%%
CPI_Energy_ratio = (CPI_Energy_usd/CPI_Energy_usd.max())-(CPI_Energy_eur/CPI_Energy_eur.max())
CPI_Energy_ratio = CPI_Energy_ratio/CPI_Energy_ratio.max()
CPI_Energy_ratio.index = CPI_Energy_ratio.index.to_period('M')
CPI_Energy_ratio = pd.DataFrame(CPI_Energy_ratio, columns = ['CPI_Energy_ratio'])

PCEPI_Roll = pd.DataFrame(PCEPI.diff(periods = 12), columns = ['PCEPI'])
PCEPI_Roll.index = PCEPI_Roll.index.to_period('M')

PCEPI_core.index = PCEPI_core.index.to_period('M')
PCEPI_core = pd.DataFrame(PCEPI_core, columns = ['PCEPI_core'])

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


#%%

cpi_usd_M = cpi_usd.copy()
cpi_usd_M.index = cpi_usd.index.to_period('M')

cpi_eur_M = cpi_eur.copy()
cpi_eur_M.index = cpi_eur.index.to_period('M')

cpi_ratio = pd.DataFrame(cpi_usd_M/cpi_usd_M.max()-cpi_eur_M/cpi_eur_M.max(), columns = ['cpi_ratio'])

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

#find rate ratio
rate_ratio = pd.DataFrame(((rate_eur/rate_usd).dropna()/10).loc[year:], columns = ['rate_ratio'])
rate_ratio.index = rate_ratio.index.to_period('M')
rate_ratio = rate_ratio.groupby(rate_ratio.index).first()

rate_dif = pd.DataFrame(((rate_eur-rate_usd*1.25).dropna()).loc[year:], columns = ['rate_dif'])
rate_dif.index = rate_dif.index.to_period('M')
rate_dif = rate_dif.groupby(rate_dif.index).first()

SP500_per_gdp = pd.concat([data_SP500_usd_M, gdp_usd_M], axis = 1).ffill()
SP500_per_gdp = SP500_per_gdp['SP500_usd']/SP500_per_gdp['gdp_usd']

FTSE_per_gdp = pd.concat([data_FTSE_eur_M, gdp_eur_M], axis = 1).ffill()
FTSE_per_gdp = FTSE_per_gdp['FTSE_eur']/FTSE_per_gdp['gdp_eur']

SP500_FTSE_ratio = pd.DataFrame(FTSE_per_gdp/SP500_per_gdp, columns = ['SP500_FTSE_ratio'])

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

asset_ratio = pd.DataFrame((assets_eur_slice/assets_usd_slice), columns = ['asset_ratio'])

#slice currency pair data
pair_data = pd.DataFrame()
pair_data['pair'] = data_M['Close'].loc[year:]#((data_M['Open'] + data_M['Close'])/2).loc[year:]

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
start_date = 1999 #'2014-06'
end_date = '2022-06'
start_date = str(start_date)
end_date = str(end_date)
test = pd.concat([ gdp_ratio/gdp_ratio.max(), m3_ratio], axis = 1).ffill()
test = pd.DataFrame(test.mean(axis = 1), columns = ['test'])

plot_df = pd.concat([exports_ratio, PCE_diff, CPI_Energy_ratio, SP500_FTSE_ratio, assets_per_gdp_diff, asset_ratio, m3_ratio, pair_data, rate_dif/rate_dif.max(), rate_usd_M, rate_eur_M, gdp_ratio/gdp_ratio.max(), test, m1_ratio, cpi_ratio*-1], axis = 1).interpolate()
ax = plt.figure()
plot_df = plot_df.loc[start_date:end_date].dropna()
#plot_df['asset_ratio'].plot(legend = True)
#(plot_df['m3_ratio']+plot_df['assets_per_gdp_diff']).plot()
#plot_df['m3_ratio'].plot()
#plot_df['m1_ratio'].plot()
#(plot_df['rate_dif'].loc[year:]*0.25).plot()
#plot_df['gdp_ratio'].plot()
#(plot_df['m3_ratio']/plot_df['gdp_ratio']).plot()
#plot_df['test'].plot()
#(plot_df['rate_usd_M'].loc[year:]/10+1).plot()
#(plot_df['rate_eur_M'].loc[year:]/10+1).plot()
comparitor = plot_df['CPI_Energy_ratio']
comparitor = comparitor/comparitor.max()
#comparitor.plot()
"""
weights = {'PCE_diff': 0,
           'CPI_Energy_ratio': 16,
           'assets_per_gdp_diff': 0,
           'rate_dif': 4,
           'cpi_ratio': -8,
           'm3_ratio': 12,
           'gdp_ratio': 0,
           'exports_ratio': 10
           }
"""

weights = {'PCE_diff': 0,
           'CPI_Energy_ratio': 1*0,
           'assets_per_gdp_diff': 0,
           'rate_dif': 0.5*0,
           'cpi_ratio': -2*0,
           'm3_ratio': 0,
           'gdp_ratio': 10*0,
           'exports_ratio': 1
           }

plot_df['net'] = ((plot_df['PCE_diff']*weights['PCE_diff'] 
                + plot_df['CPI_Energy_ratio']*weights['CPI_Energy_ratio']  
                + plot_df['assets_per_gdp_diff']*weights['assets_per_gdp_diff'] 
                + plot_df['rate_dif'].loc[year:]*weights['rate_dif']  
                + plot_df['cpi_ratio']*weights['cpi_ratio'] 
                + plot_df['m3_ratio']*weights['m3_ratio'] 
                + plot_df['gdp_ratio']*weights['gdp_ratio']
                + plot_df['exports_ratio']*weights['exports_ratio']))
plot_df['net'] = plot_df['net']*((plot_df['net'].max()-plot_df['net'].min())/(plot_df['pair'].max()-plot_df['pair'].min()))**-1
plot_df['net'] = plot_df['net'] + (plot_df['pair'].mean() - plot_df['net'].mean() )
plot_df['m3_ratio'] = plot_df['m3_ratio']*((plot_df['m3_ratio'].max()-plot_df['m3_ratio'].min())/(plot_df['pair'].max()-plot_df['pair'].min()))**-1
plot_df['m3_ratio'] = plot_df['m3_ratio'] + (plot_df['pair'].mean() - plot_df['m3_ratio'].mean() )

ax = plot_df[['pair']].plot(color = ['g'], style = ['-'])

#ax2 = ax1.twinx()
#ax2.spines['right'].set_position(('axes', 1.0))
roll = 4
shift = 0
diff_threshold = 0.01
stop = -0.05
diff_range = 4

trade_on = 'pair'
#trade_on = 'net'

#plot_df['net_sgn'] = np.sign(plot_df[[trade_on]].rolling(roll).mean().diff().shift(shift))
plot_df['net_sgn'] = np.sign(plot_df[[trade_on]].rolling(roll).mean().diff(diff_range).diff(diff_range).shift(shift))
plot_df['net_sgn'][plot_df[trade_on].rolling(roll).mean().diff(diff_range).diff(diff_range).shift(shift).abs() < diff_threshold] = float('NaN')
plot_df['net_sgn'] = plot_df['net_sgn'].ffill().dropna()


(plot_df[['net_sgn']]/8+1.3).plot(ax = ax, color = ['b', 'r'], style = ['-o', '-'])
plot_df[[trade_on]].rolling(roll).mean().shift(0).plot(ax = ax, color = ['m'], style = ['--'])
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
ax.legend()
plot_df['net_sgn'].iloc[-1] = plot_df['net_sgn'].iloc[-1]*-1
trades = ((plot_df['net_sgn'].dropna().diff()/-2)*plot_df['pair'])
short = trades[trades > 0].reset_index().iloc[:,1]
long = trades[trades < 0].reset_index().iloc[:,1]
#new_index = (trades[trades < 0].index)
trades_net = pd.concat([short, long], axis = 1, ignore_index = True)
#trades_net = trades_net.set_index(new_index)
trades_net = trades_net.dropna().sum(axis = 1)
trades_net[trades_net < stop] = stop
total = trades_net.sum()
print(total)
#trades_net.cumsum().plot()
plt.figure()
trades_net.cumsum().plot()

    #%%

PCEPI_Food_Energy =  pd.concat([PCEPI_core, PCEPI_Roll], axis = 1)
PCEPI_Food_Energy = pd.DataFrame(PCEPI_Food_Energy['PCEPI'] -  PCEPI_Food_Energy['PCEPI_core'], columns = ['PCEPI_Food_Energy'])
PCEPI_Food_Energy.plot()
