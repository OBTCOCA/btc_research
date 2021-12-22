# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from glassnode import *
from statsmodels.tsa.stattools import adfuller
from stqdm import stqdm

import statsmodels.api as sm
from scipy.stats import mode

sns.set()

# %%
def get_glassnode_price():
    GLASSNODE_API_KEY = '1vUcyF35hTk9awbNGszF0KcLuYH'

    self = GlassnodeClient()
    self.set_api_key(GLASSNODE_API_KEY)

    url = URLS['Market'] + 'price_usd_ohlc'
    a ='BTC'
    c = 'native'
    i='24h'

    ohlc = self.get(url,a,i,c)
    return ohlc
# %%
ohlc = get_glassnode_price()
ohlc = ohlc.rename(columns = {'c':'close','h':'high','l':'low','o':'open'})
# %%
Px = ohlc[['high','low']].mean(axis = 1).rename('price')

slow_ma = Px.rolling(220).mean().rename('slow_ma')
fast_ma = Px.rolling(5).mean().rename('fast_ma')

price = pd.concat([Px,slow_ma,fast_ma],axis = 1)

price = price.dropna()
# %%
f = go.Figure()
f.add_trace(go.Scatter(x=price.index, y=price['price'],
                    mode='lines',
                    name='BTC'))

f.add_trace(go.Scatter(x=price.index, y=price['fast_ma'],
                    mode='lines',
                    name='fast MAV'))

f.add_trace(go.Scatter(x=price.index, y=price['slow_ma'],
                    mode='lines',
                    name='slow MAV'))

f.update_yaxes(type="log")

f.update_layout(legend=dict(
yanchor="top",
y=0.99,
xanchor="left",
x=0.01))

# %%
price['ma_spread'] = price['fast_ma']-price['slow_ma']
price['ma_spread_sign'] = np.sign(price['ma_spread'])
price['diff_ma_spread_sign'] = price['ma_spread_sign'].diff()
price.dropna(inplace = True)
# %%
signals = price.loc[price.diff_ma_spread_sign.abs()==2].copy()
signals['mult'] = signals['diff_ma_spread_sign'].shift(-1)/2 
signals['R'] = np.log(signals).price.diff()
# %%
(signals.R*signals.mult).cumsum().dropna().plot()
# %%
