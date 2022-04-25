# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import statsmodels.api as sm 
from tqdm import tqdm
import sys
import itertools

from smoothers import simple_smooth
from BandPass import BandPass,TruncatedBandPass

plt.rcParams['figure.figsize'] = [18, 10]

# %%
path = '/Users/orentapiero/Data/'
pair = 'usdcad'

ohlc=pd.read_csv(path+pair+'_5T_'+'ohlc.csv')
ohlc.index = pd.to_datetime(ohlc['TIMESTAMP'])
    
del ohlc['TIMESTAMP']
    
Mid = ohlc[['mid_open','mid_high','mid_low','mid_close']]
Ask = ohlc[['ask_open','ask_high','ask_low','ask_close']]
Bid = ohlc[['bid_open','bid_high','bid_low','bid_close']]

Ask = Ask.groupby(Ask.index).last()
Bid = Bid.groupby(Bid.index).last()
Mid = Mid.groupby(Mid.index).last()
# %%
x = Mid.mid_close
# %%
x1 = x.loc['2016-02-26']
x2 = simple_smooth(x).loc['2016-02-26']

bp1 = BandPass(x1,88,0.95)
bp2 = BandPass(x2,88,0.95)

zer = pd.Series(0,index = bp1.index)

f,a = plt.subplots(nrows = 2,sharex = True)
x1.plot(ax = a[0])
x2.plot(ax = a[0])
bp1.plot(ax = a[1])
bp2.plot(ax = a[1])
zer.plot(ax = a[1],style = 'k--')


# %%
x1 = x.fillna(method = 'ffill').loc['2014-03-31']

bp1 = BandPass(x1,88,0.95)
bp2 = TruncatedBandPass(x1,88,0.95,1*12)
w = 0.85


zer = pd.Series(0,index = bp1.index)

f,a = plt.subplots(nrows = 2,sharex = True)
x1.between_time('09:00','15:30').plot(ax = a[0])
bp1.between_time('09:00','15:30').plot(ax = a[1])

(w*bp1+(1-w)*bp2).between_time('09:00','15:30').plot(ax = a[1], style = 'm--')
zer.between_time('09:00','15:30').plot(ax = a[1],style = 'k--')


# %%
