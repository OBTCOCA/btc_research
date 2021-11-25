# %%
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import statsmodels.api as sm

from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller

from glassnode import *
from BandPass import BandPass
from smoothers import strided_app,VWAP
from SVR_FN import SVR_predictor,forecast_ols_evaluation,regression_cm

sns.set()
plt.rcParams['figure.figsize'] = [15, 7]
# %%
def hpwrap(x,pars):
    _,trend = sm.tsa.filters.hpfilter(x,pars)
    return trend
# %%
Addresses = ['count', 'sending_count','receiving_count', 
             'active_count','non_zero_count', 'min_1_count',
             'min_10_count', 'min_100_count','min_1k_count', 
             'min_10k_count']

Blockchain = ['utxo_created_count', 'utxo_created_value_sum',
              'utxo_spent_value_sum', 'utxo_created_value_mean',
              'utxo_created_value_median','utxo_spent_value_median', 
              'utxo_profit_count','utxo_loss_count', 
              'utxo_profit_relative','block_height', 
              'block_count','block_interval_median']

Distribution = ['exchange_net_position_change','balance_1pct_holders', 
                'gini']

Indicators = ['rhodl_ratio', 'balanced_price_usd','difficulty_ribbon_compression',
               'nvt','nvts', 'cdd_supply_adjusted_binary','average_dormancy_supply_adjusted',
               'reserve_risk', 'cyd','cdd90_age_adjusted', 'sopr', 'asol','msol', 'unrealized_profit',
               'unrealized_loss', 'nupl_less_155','nupl_more_155', 'dormancy_flow',
               'net_realized_profit_loss','realized_profit_loss_ratio','stock_to_flow_deflection', 'realized_loss',
               'sol_1h', 'sol_1h_24h', 'sol_1d_1w','sol_1w_1m', 'sol_1m_3m', 'sol_3m_6m','sol_6m_12m', 
               'sol_1y_2y', 'sol_2y_3y','sol_7y_10y']

Market = ['price_drawdown_relative','deltacap_usd', 'marketcap_usd', 'mvrv','mvrv_z_score']

Mining = ['difficulty_latest','revenue_from_fees', 'marketcap_thermocap_ratio']

Supply = ['current', 'issued', 'inflation_rate','active_24h', 'active_1d_1w', 
          'active_1w_1m','active_1m_3m', 'active_3m_6m', 'active_6m_12m',
          'active_1y_2y', 'active_2y_3y','active_more_1y_percent',
           'active_more_3y_percent']

Transactions = ['size_mean', 'size_sum','transfers_volume_adjusted_sum',
                'transfers_volume_adjusted_median',
                'transfers_volume_from_exchanges_mean',
                'transfers_volume_exchanges_net',
                'transfers_to_exchanges_count',
                'transfers_from_exchanges_count']

urls = []

for a in Addresses:
    urls += [URLS['Addresses']+a]

for b in Blockchain:
    urls += [URLS['Blockchain']+b]

for d in Distribution:
    urls += [URLS['Blockchain']+d]

for i in Indicators:
    urls += [URLS['Indicators']+i]

for m in Mining:
    urls += [URLS['Mining']+m]

for s in Supply:
    urls += [URLS['Supply']+s]

for t in Transactions:
    urls += [URLS['Transactions']+t]

for m1 in Market:
    urls += [URLS['Market']+m1]

# %% Get Price Data

GLASSNODE_API_KEY = '1vUcyF35hTk9awbNGszF0KcLuYH'

self = GlassnodeClient()
self.set_api_key(GLASSNODE_API_KEY)

url = URLS['Market'] + 'price_usd_ohlc'
a ='BTC'
c = 'native'
i='24h'

ohlc = self.get(url,a,i,c)
# %%
price = ohlc['c']
gr = price.groupby(price.index.strftime('%Y-%m'))
denoised_price = gr.apply(lambda x: hpwrap(x,10))

# %%
price.loc['2018'].plot()
denoised_price.loc['2018'].plot()

# %% Get features data

GLASSNODE_API_KEY = '1vUcyF35hTk9awbNGszF0KcLuYH'

self = GlassnodeClient()
self.set_api_key(GLASSNODE_API_KEY)

features = []

for u in tqdm(urls):
    a,c,i='BTC','native','24h'
    z = self.get(u,a,i,c)
    
    try:
        features.append(z.rename(u.split('/')[-1]))
    except:
        message = f"cannot get {u.split('/')[-1]}."
        print(message)

features = pd.concat(features,axis = 1)
features = features.loc['2013':]
# %% check for integrated series

processed_features = []

for c in tqdm(features.columns):
    series = features[c]
    ser = features[c].astype('float32').copy()
    
    Nd = str(ser.abs().mean()).split('.')[0]
    Ndigits = len(list(Nd))
    
    if Ndigits >1:
        divisor = eval(str('1e')+str(Ndigits))/10
        ser = ser/divisor
    
    adf = adfuller(ser, maxlag=5, regression='ct')
    
    if adf[1] >= 0.1:
        if ser.min() > 0:
            processed_features.append(100*np.log(ser).diff().rename(c))
        else:
            processed_features.append(100*ser.diff().rename(c))
    else:
        processed_features.append(ser.rename(c))
  
processed_features = pd.concat(processed_features,axis = 1)

# %% Create Target/Features data frame

Target = 100*np.log(denoised_price.shift(-1)/denoised_price).rename('Target')
Pseudo = 100*np.log(price.shift(-1)/price).rename('PTarget')

xcols = ['sopr',
         'msol',
         'marketcap_thermocap_ratio',
         'nvts',
         'realized_profit_loss_ratio',
         'rhodl_ratio',
         'marketcap_thermocap_ratio']

Xdf = processed_features[xcols]

df = pd.concat([Target,Xdf],axis = 1).dropna()
# %%
Months = np.unique(df.index.strftime('%Y-%m'))
StridedMonths = strided_app(Months,36,1)

frequency = '%Y-%m'
Target = ['Target']

Features = Xdf.columns

kernel = 'linear'

Y = []
Prc = []

for i in tqdm(range(StridedMonths.shape[0])):
  trainPeriod = list(StridedMonths[i][:-1])
  cvPeriod = StridedMonths[i][-1]

  self = SVR_predictor(df,trainPeriod,cvPeriod,frequency,Target,Features,kernel)
  self.train_cv_split() 
  self.fit()
  Y_ = self.predict()
  Y.append(Y_)
  Prc.append({'month':cvPeriod,'prc':regression_cm(Y_)})

Y = pd.concat(Y,axis = 0)
# %%
out = forecast_ols_evaluation(Y['Target'],Y['estimated'])
X = sm.add_constant(Y['estimated'])
y = Y['Target']
mod = sm.OLS(y,X).fit()

mod.summary()
# %%
Y.cumsum().plot(figsize = (15,7))
# %%
print('pseudo precision:',f'{100*regression_cm(Y)}%')
# %%
plt.scatter(Y.estimated,Y.Target)
