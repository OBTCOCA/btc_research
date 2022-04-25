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
from cycles import simple_cycle,simple_quadrature
from utilities import strided_app

plt.rcParams['figure.figsize'] = [18, 10]

# %%
path = '/Users/orentapiero/Data/'
pair = 'eurusd'

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

bp1 = BandPass(x1,98,0.95)
bp2 = BandPass(x1,98,0.95)

zer = pd.Series(0,index = bp1.index)

f,a = plt.subplots(nrows = 2,sharex = True)
x1.plot(ax = a[0])
x2.plot(ax = a[0])
bp1.plot(ax = a[1])
bp2.plot(ax = a[1])
zer.plot(ax = a[1],style = 'k--')


# %%
x1 = x.fillna(method = 'ffill').loc['2014-08-27']
x2 = simple_smooth(x.fillna(method = 'ffill')).loc['2014-08-27']

bp1 = BandPass(x2,88,0.95)
bp2 = TruncatedBandPass(x2,88,0.95,36)
w = 0.75


zer = pd.Series(0,index = bp1.index)

f,a = plt.subplots(nrows = 2,sharex = True)
x1.plot(ax = a[0])
bp1.plot(ax = a[1])
bp2.plot(ax=a[1])
(w*bp1+(1-w)*bp2).plot(ax = a[1], style = 'm--')
zer.plot(ax = a[1],style = 'k--')


# %%
x1 = x.fillna(method = 'ffill').loc['2014-03-31']
x2 = simple_smooth(x.fillna(method = 'ffill')).loc['2014-03-31']

bp1 = BandPass(x1,102,0.95)
bp2 = TruncatedBandPass(x2,102,0.95,1*12)
w = 0.85


zer = pd.Series(0,index = bp1.index)

f,a = plt.subplots(nrows = 3,sharex = True)
x1.plot(ax = a[0])
bp1.plot(ax = a[1])
zer.plot(ax = a[1],style = 'k--')
bp1.diff(7).plot(ax = a[2])
zer.plot(ax = a[2],style = 'k--')

# %%
x1 = x.fillna(method = 'ffill').loc['2014-03-31']
x2 = simple_smooth(x.fillna(method = 'ffill')).loc['2014-03-31']

c=simple_cycle(x2,0.05)
q=simple_quadrature(x2)

# %%
f,a = plt.subplots(nrows = 2,sharex = True)
x1.plot(ax = a[0])
x2.plot(ax = a[0])

#bp1.plot(ax = a[1])
q.plot(ax = a[1])

# %%

# %%
P = x.groupby(x.index.date).last()
P.index = pd.to_datetime(P.index)
P=P

# %%
r = 10000*np.log(P/P.shift(1)).dropna()

L= 66
R = (r - r.rolling(L).mean())/r.rolling(L).std() 
R=R.dropna()

# %%
M=np.sqrt(1-np.exp(-4/66))
e=np.exp(-2*np.arange(66)/66)
w = M*e[::-1]


Rs = strided_app(R.values,len(w),1)
psi =np.empty_like(R.values)

arr = [(Rs[i,:]*w).sum() for i in range(Rs.shape[0])]

psi[(len(w)-1):]=arr

psi = pd.Series(psi,index = R.index)

# %%
f,a = plt.subplots(nrows = 2,sharex = True)
P.loc[psi.index].loc['2016':].plot(ax = a[0])
psi.loc['2016':].plot(ax = a[1])
pd.Series(0,index = psi.index).loc['2016':].plot(ax=a[1],style = 'k--')

# %%
r=r.loc[psi.index]

# %%
r[(psi.shift(1)>=0)].cumsum().plot()
(-r[(psi.shift(1)<=0)]).cumsum().plot()

rs = r[(psi.shift(1)>=0)]

# %%
rs

# %%
rs[rs>=0].count()/len(rs)

# %%
print(rs[rs>0].mean())
print(rs[rs<0].mean())

# %%
rs = r[(psi.rolling(3).mean().shift(1)<=0)]

# %%
rs[rs<0].count()/len(rs)

# %%
psi

# %%
z = x1

np.polyfit(np.arange(len(z)),z.values,2)

# %%
z = np.log(x1)+np.log(x1).diff().expanding().mean()

# %%
np.log(x1).plot()
z.plot()


# %%
x1.plot()
x1.rolling(5).median().plot()

# %%
