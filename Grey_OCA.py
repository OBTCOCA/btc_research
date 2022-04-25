# %%
import sys 

sys.path.insert(0,'/Users/orentapiero/Documents/MyResearch/')

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpl
import math
import statsmodels.tsa.stattools as ts 
import itertools

from tqdm import tqdm
from FILTERS.utilities import strided_app
from FILTERS.hpfilter import hprescott
from FILTERS.HPindicators import hp_avg_trend_bps,roll_hp_trend
from FILTERS.BandPass import BandPass

from btc_functions.glassnode import *
from btc_functions.import_data import get_glassnode_price,get_glassnode_data
from btc_functions.variable_list_urls import *
from btc_functions.utilities import strided_app,strided_app2

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.vector_ar import vecm

from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
set_loky_pickler()

from backtesting import Backtest, Strategy

plt.rcParams['figure.figsize'] = [15, 10]
sns.set()

# %%

def GM11(x,n):
    x1 = x.cumsum()
    z1 = (0.5*x1[:(len(x1)-1)] + 0.5*x1[1:])
    z1 = z1.reshape((len(z1),1))

    B = np.append(-z1,np.ones_like(z1),axis=1)
    Y = x[1:].reshape((len(x)-1,1))
    [[a],[b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T,B)),B.T),Y)
    result = (x[0]-b/a)*np.exp(-a*(n-1))-(x[0]-b/a)*np.exp(-a*(n-2))
    S1_2 = x.var()

    e = list()
    fy = list()

    for index in range(1,x.shape[0]+1):
        predict = (x[0]-b/a)*np.exp(-a*(index-1))-(x[0]-b/a)*np.exp(-a*(index-2))
        e.append(x[index-1]-predict)
        fy.append(predict)

    S2_2 = np.array(e).var()
    C = S2_2/S1_2

    predict = list()
    for index in range(x.shape[0]+1,x.shape[0]+n+1):
            predict.append((x[0]-b/a)*np.exp(-a*(index-1))-(x[0]-b/a)*np.exp(-a*(index-2)))
    predict = np.array(predict)
    
    return {'curr':x[-1],'pred': predict[-1],'acc':C}

def FGM1(x,l=2,n=1):
    x1 = x.cumsum()
    z1 = (0.5*x1[:(len(x1)-1)] + 0.5*x1[1:])
    z1 = z1.reshape((len(z1),1))

    B = np.append(-z1,np.ones_like(z1),axis=1)
    B = B[:-(l-1),:]
    Y = x[l:].reshape((len(x)-l,1))

    [[a],[b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T,B)),B.T),Y)
    S1_2 = x.var()

    e = list()
    fy = list()
    f = lambda k: (x[0]-(b/a))*(1-math.exp(a))*math.exp(-a*(k-1))

    for index in range(1,x.shape[0]+1):
        predict = f(index)
        e.append(x[index-1]-predict)
        fy.append(predict)

    S2_2 = np.array(e).var()
    C = S2_2/S1_2

    predict = list()
    for index in range(x.shape[0]+1,x.shape[0]+n+1):
            predict.append(f(index))
    predict = np.array(predict)
    
    return {'curr':x[-1],'pred': predict[-1],'acc':C}

def GMWrapper(Z,dates,n = 1):
    x = Z.loc[dates].values
    predict,C = GM11(x,n)
    return {'date':dates[-1],'pred':predict[-1],'accuracy':C}

def strided_GM11(Price,L,n=1):
    Px = strided_app(Price.values,L,1)
    T = Px.shape[0]
    grey = len(Price)*[{'pred':np.nan,'acc':np.nan}]
    tmp = [GM11(Px[t,:],n) for t in tqdm(range(T))]
    grey[(L-1):] = tmp
    grey = pd.DataFrame(grey)
    grey.index = Price.index
    return grey

def strided_FGM1(Price,L,l=2,n=1):
    Px = strided_app(Price.values,L,1)
    T = Px.shape[0]
    grey = len(Price)*[{'pred':np.nan,'acc':np.nan}]
    tmp = [FGM1(Px[t,:],l,n) for t in tqdm(range(T))]
    grey[(L-1):] = tmp
    grey = pd.DataFrame(grey)
    grey.index = Price.index
    return grey

def resample_ohlc(df,freq):
    gr = df.resample(freq,closed='right',label='right')
    Op = gr.Open.first()
    Hi = gr.High.max()
    Lo = gr.Low.min()
    Cl = gr.Close.last()

    X = pd.concat([Op,Hi,Lo,Cl],axis = 1)
    Y = X.reset_index().copy()

    Y.loc[Y.index[-1],'Time'] = df.index[-1]
    Y.index = pd.to_datetime(Y['Time'])
    del Y['Time']
    return Y

def gen_fn(x,alpha,beta,mu):
    if alpha*alpha < 4*beta*mu:
        Lambda = math.sqrt(4*beta*mu - alpha**2)
        Phi = (1-2/Lambda)*math.atan((2*beta*x[0]-alpha)/Lambda)

        f = lambda k: (Lambda/(2*beta))*math.tan((Lambda*(k-Phi))/2) + alpha/(2*beta)
    elif alpha*alpha > 4*beta*mu:
        if beta>0:
            Lambda = math.sqrt(alpha**2 - 4*beta*mu)
            phi_up = 2*beta*x[0]-alpha-Lambda
            phi_dn = 2*beta*x[0]-alpha+Lambda
            Phi = 1-(1/Lambda)*math.log(abs(phi_up/phi_dn))

            f = lambda k: (alpha/(2*beta)) - (Lambda/(2*beta))+(1+2/(math.exp(Lambda*(k-Phi))-1))
        elif beta < 0:
            Lambda = math.sqrt(alpha**2 - 4*beta*mu)
            phi_up = 2*beta*x[0]-alpha-Lambda
            phi_dn = 2*beta*x[0]-alpha+Lambda
            Phi = 1-(1/Lambda)*math.log(abs(phi_up/phi_dn))

            f = lambda k: (alpha/(2*beta)) - (Lambda/(2*beta))+(1-2/(math.exp(Lambda*(k-Phi))+1))
    elif (alpha == 2*math.sqrt(beta*mu)):
        Phi = 1+1/(beta*x[0]-math.sqrt(beta*mu))
        f = lambda k: (1/beta)*(math.sqrt(beta*mu)-1/(k-Phi))
    elif (alpha == -2*math.sqrt(beta*mu)):
        Phi = 1+1/(beta*x[0]+math.sqrt(beta*mu))
        f = lambda k: (1/beta)*(math.sqrt(beta*mu)+1/(k-Phi))
    
    return f

def GVM(x,n):
    x1 = x.cumsum()
    z1 = (0.5*x1[:(len(x1)-1)] + 0.5*x1[1:])
    z1 = z1.reshape((len(z1),1))
    z1_sq = z1**2
    B = np.append(-z1,z1_sq,axis=1)
    B = np.append(B,np.ones_like(z1),axis = 1)
    Y = x[1:].reshape((len(x)-1,1))
    [[alpha],[beta],[mu]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T,B)),B.T),Y)
    f = gen_fn(x,alpha,beta,mu)

    fy = list()

    for index in range(1,x.shape[0]+1):
        predict = f(index)
        fy.append(predict)


    dfy = np.diff(fy,prepend=np.nan)
    dfy[0] = x[0]

    e = x[1:]-dfy[1:]
    S2_2 = np.array(e).var()
    S1_2 = x[1:].var()

    C = S2_2/S1_2

    predict = list()
    for index in range(x.shape[0]+1,x.shape[0]+n+1):
            predict.append(f(index)-f(index-1))
    predict = np.array(predict)
    return {'curr':x[-1],'pred': predict[-1],'acc':C}

def strided_GVM(Price,L,n=1):
    Px = strided_app(Price.values,L,1)
    T = Px.shape[0]
    grey = len(Price)*[{'pred':np.nan,'acc':np.nan}]
    tmp = [GVM(Px[t,:],n) for t in tqdm(range(T))]
    grey[(L-1):] = tmp
    grey = pd.DataFrame(grey)
    grey.index = Price.index
    return grey

# %%

Urls['transfers_volume_sum'] = 'https://api.glassnode.com/v1/metrics/transactions/transfers_volume_sum'
Urls['ssr'] = 'https://api.glassnode.com/v1/metrics/indicators/ssr'


ohlc = get_glassnode_price()
ohlc = ohlc.rename(columns = {'o':'Open','h':'High','l':'Low','c':'Close'})
ohlc = ohlc.loc[:'2019']

selected = ['marketcap_usd',
            'mvrv_z_score',
            'sopr_adjusted',
            'puell_multiple',
            'net_unrealized_profit_loss',
            'transfers_volume_sum',
            'transfers_volume_exchanges_net',
            'dormancy_flow',
            'reserve_risk',
            'cdd90_age_adjusted',
            'average_dormancy',
            'liveliness', 
            'realized_profits_to_value_ratio',
            'rhodl_ratio',
            'cvdd',
            'nvts',
            'marketcap_thermocap_ratio']

raw_features = get_glassnode_data(selected,Urls)
raw_features = raw_features.loc[:'2019']

# %%
ohlc = ohlc.loc['2013':]

log_rhodl = np.log(raw_features.rhodl_ratio)
mvrv_z_score = raw_features.mvrv_z_score
log_marketcap = np.log(raw_features.marketcap_usd)
sopr_adjusted = raw_features.sopr_adjusted
log_puell_multiple = np.log(raw_features.puell_multiple)
net_unrealized_profit_loss = raw_features.net_unrealized_profit_loss
log_transfers_volume_sum = np.log(raw_features.transfers_volume_sum)
transfers_volume_exchanges_net = raw_features.transfers_volume_exchanges_net
log_dormancy_flow = np.log(raw_features.dormancy_flow)
log_reserve_risk = np.log(raw_features.reserve_risk)
log_cdd90_age_adjusted = np.log(raw_features.cdd90_age_adjusted)
log_average_dormancy = np.log(raw_features.average_dormancy)
log_liveliness = np.log(raw_features.liveliness)
log_realized_profits_to_value_ratio = np.log(raw_features.realized_profits_to_value_ratio)
log_cvdd = np.log(raw_features.cvdd)
log_nvts = np.log(raw_features.nvts)
log_marketcap_thermocap_ratio = np.log(raw_features.marketcap_thermocap_ratio)


features = pd.concat([log_rhodl,mvrv_z_score,log_marketcap,
                      sopr_adjusted,log_puell_multiple,net_unrealized_profit_loss,
                      log_transfers_volume_sum,transfers_volume_exchanges_net,
                      log_dormancy_flow,log_reserve_risk,log_cdd90_age_adjusted,
                      log_average_dormancy,log_liveliness,
                      log_realized_profits_to_value_ratio,
                      log_cvdd,log_nvts,log_marketcap_thermocap_ratio],axis = 1)

px_cl = np.log(ohlc.Close)
px_m1 = np.log(ohlc[['High','Low']].mean(1))
px_m2 = np.log(ohlc[['Open','Close']].mean(1))
R = 100*px_cl.diff().rename('return')
#%% Cointegration test with price

res = []
for c in features.columns: 
    result=ts.coint(features[c], px_cl)
    out = dict(var = c,pval=result[1])
    res.append(out)

coint_res = pd.DataFrame(res).sort_values(by = 'pval',ascending=False)
coint_res


# %% Cointegration test with variables
featcols = list(features.columns)
combos = [",".join(map(str, comb)) for comb in itertools.combinations(featcols, 2)]

res = []
for comb in combos:
    n1,n2 = comb.split(',')
    x,y = features[n1],features[n2]
    result=ts.coint(x,y)
    out = dict(var1 = n1,var2 = n2,pval=result[1])
    res.append(out)

cx_coint_res = pd.DataFrame(res).sort_values(by = 'pval',ascending=False)
cx_coint_res

# %%
f_dorm = strided_GM11(features.average_dormancy,500,n=1)
f_realized_profits_to_value_ratio = strided_GM11(features.rhodl_ratio,500,n=1)
# %%
from sklearn.preprocessing import MinMaxScaler


BP_grey_features = []
cols2BP = coint_res.loc[coint_res.pval > 0.15,'var'].values.tolist()

for c in cols2BP:
    f_ser = strided_GM11(features[c],500,n=1).dropna()
    del_f_ser = BandPass(f_ser['pred'],90,0.9,2).rename(c)
    BP_grey_features.append(del_f_ser)

BP_grey_features = pd.concat(BP_grey_features,axis = 1) 

# %%
res = []
for c in BP_grey_features.columns:
    df = pd.DataFrame()
    df = pd.concat([R,BP_grey_features[c].shift(1)],axis = 1).dropna()
    pos = df.loc[df[c]>0,'return'].mean()
    neg = df.loc[df[c]<0,'return'].mean()
    adpos = pos/df.loc[df[c]>0,'return'].std()
    adneg = neg/df.loc[df[c]<0,'return'].std()

    prc_pos = df.loc[(df[c]>0) & (df['return']>0),'return'].count()/df.loc[df[c]>0,'return'].count()
    prc_neg = df.loc[(df[c]<0) & (df['return']>0),'return'].count()/df.loc[df[c]<0,'return'].count()

    out = {'var':c, 
           'pos':pos,
           'neg':neg,
           'adpos':adpos,
           'adneg':adneg,
           'prc_pos':prc_pos,
           'prc_neg':prc_neg}

    res.append(out)

res = pd.DataFrame(res)
res



# %%

def Indicator(series):
    return series

class indicatorWave(Strategy):

    def init(self):
        variable = self.data.variable
        self.variable = self.I(Indicator,variable)
        
    
    def next(self):

        if (self.variable > 0.001) and not (self.position.is_long):
            self.buy(size=.2)
        elif (self.variable < -0.001) and not (self.position.is_short):
            self.sell(size=.2)


res = []
for c in BP_grey_features.columns:
    ohlc_df = ohlc.copy()
    ohlc_df['variable'] = BP_grey_features[c]
    ohlc_df = ohlc_df.dropna()

    bt = Backtest(ohlc_df, indicatorWave,
                cash=1000000, commission=.002,
                exclusive_orders=True)

    output = bt.run()

    out = {'variable':c,
           'Return':output.loc['Return [%]'],
           'Sharpe': output.loc['Sharpe Ratio'],
           '# Trades':output.loc['# Trades'],
           'Win Rate [%]':output.loc['Win Rate [%]'],
           'Expectancy [%]': output.loc['Expectancy [%]']}


    res.append(out)

pd.DataFrame(res).sort_values(by = 'Return',ascending=False)

# %%
ohlc_df = ohlc.copy()
ohlc_df['variable'] = BP_grey_features['cdd90_age_adjusted']

ohlc_df = ohlc_df.dropna()

bt = Backtest(ohlc_df, indicatorWave,
            cash=1000000, commission=.002,
            exclusive_orders=True)

output = bt.run()
bt.plot()

# %%
