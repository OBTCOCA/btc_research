# %%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mplfinance as mpf
import sys

from tqdm import tqdm

sys.path.insert(0,'/Users/orentapiero/btc_research/')

from btc_functions.utilities import strided_app,strided_app2
from btc_functions.wavelet_transform import WT
from btc_functions.HPindicators import roll_hp_trend
sns.set()

plt.rcParams['figure.figsize'] = [18, 10]
sns.set()
# %%
path = '/Users/orentapiero/Data/BTCIntraday/'
ticks = []

for y in ['2017','2018','2019','2020','2021']:
    files = os.listdir(path+y)
    for f in tqdm(files):
        try:
            data = pd.read_csv(path+y+'/'+f,header = None)
            date = data.iloc[:,0].str.replace(':','.',n=-1)
            date = date.str.replace('.',':',n=2)
            data.index = pd.to_datetime(date)
            data = data.iloc[:,1:]
            data.columns =  ['Price','Volume','N']

            period = data.index.strftime('%Y-%m-%d %H:%M')
            ohlc = data.groupby(period)['Price'].ohlc()
            volume = data.groupby(period)['Volume'].sum().rename('Volume')
            Nt = data.groupby(period)['N'].sum().rename('N')

            ohlc = pd.concat([ohlc,volume,Nt],axis = 1)
            ohlc.index = pd.to_datetime(ohlc.index)
            ohlc.index.name = 'time'
            ticks.append(ohlc)
        except:
            pass
    
ticks = pd.concat(ticks,axis=0)

# %%
ticks = ticks.groupby(ticks.index).first()
ticks.to_csv('/Users/orentapiero/Data/BTC1M.csv')
# %%
