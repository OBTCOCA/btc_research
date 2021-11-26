# +
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from urllib.error import URLError
from glassnode import *
from statsmodels.tsa.stattools import adfuller
from stqdm import stqdm

import statsmodels.api as sm
from SVR_FN import SVR_predictor,forecast_ols_evaluation,regression_cm

sns.set()

# +
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

Indicators = ['hodler_net_position_change','realized_profits_to_value_ratio','investor_capitalization','cvdd','rhodl_ratio', 'balanced_price_usd','difficulty_ribbon_compression',
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
    
Urls = dict()

for u in urls:
    Urls[u.split('/')[-1]] = u


# +
#@st.cache

def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

def hpwrap(x,pars):
    _,trend = sm.tsa.filters.hpfilter(x,pars)
    return trend

# @st.cache(show_spinner=True)
def get_glassnode_price():
    GLASSNODE_API_KEY = '1vUcyF35hTk9awbNGszF0KcLuYH'

    self = GlassnodeClient()
    self.set_api_key(GLASSNODE_API_KEY)

    url = URLS['Market'] + 'price_usd_ohlc'
    a ='BTC'
    c = 'native'
    i='24h'

    ohlc = self.get(url,a,i,c)
    price = ohlc['c']
    return price
    
# @st.cache(show_spinner=True)
def get_glassnode_data(list_variables,Urls):
    GLASSNODE_API_KEY = '1vUcyF35hTk9awbNGszF0KcLuYH'
    
    urls = []
    
    for i in list_variables:
        urls.append(Urls[i])
    
    self = GlassnodeClient()
    self.set_api_key(GLASSNODE_API_KEY)

    features = []
    
    for u in urls:
        a,c,i='BTC','native','24h'
        z = self.get(u,a,i,c)

        try:
            features.append(z.rename(u.split('/')[-1]))
        except:
            message = f"cannot get {u.split('/')[-1]}."
            print(message)

    features = pd.concat(features,axis = 1)
    features = features.loc['2013':]
    return features

# @st.cache(show_spinner=True)
def process_variables(features):
    processed_features = []
    for c in features.columns:
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
    return processed_features

# @st.cache(show_spinner=True)
def predcition_analysis(Xdf,price,period,n_train):    
    freq = np.unique(Xdf.index.strftime(period))
    StridedMonths = strided_app(freq,n_train,1)

    frequency = period
    Target = ['Target']

    Features = Xdf.columns

    kernel = 'linear'

    Y = []
    Prc = []

    for i in stqdm(range(StridedMonths.shape[0])):

        trainPeriod = list(StridedMonths[i][:-1])
        cvPeriod = StridedMonths[i][-1]
  
        train_price = price.shift(-1).loc[price.index.strftime(period).isin(trainPeriod)]
        cv_price = price.shift(-1).loc[price.index.strftime(period).isin([cvPeriod])]
        pr = pd.concat([train_price,cv_price],axis = 0).sort_index()
        
        denoised_price_train = hpwrap(np.log(train_price),10)
        denoised_price_all = hpwrap(np.log(pr),10)


        logPx = denoised_price_train
        logPx.loc[denoised_price_all.index[-1]] = denoised_price_all.iloc[-1]
        
        r = 100*logPx.diff()
        df = pd.concat([r.rename('Target'),Xdf.loc[r.index]],axis = 1).dropna()
    
        self = SVR_predictor(df,trainPeriod,cvPeriod,frequency,Target,Features,kernel)
        self.train_cv_split() 
        self.fit()
        Y_ = self.predict()
        Y.append(Y_)
        Prc.append({'month':cvPeriod,'prc':regression_cm(Y_)})

    Y = pd.concat(Y,axis = 0)
    return Y

# +
# price = get_glassnode_price()
# list_variables = ['sopr','mvrv_z_score','deltacap_usd']
# Xdf = get_glassnode_data(list_variables,Urls)
# Zdf = process_variables(Xdf)
# Y = predcition_analysis(Zdf,price,'%Y-%m-%d',120)

# +
# 
try:
    list_variables = st.multiselect("Choose on chain indicators", list(Urls.keys()), ["nvt", "msol"])
    train_days = st.slider('Number of days to train?', 10,252)

    if not list_variables:
        st.error("Please select at least one indicators.")
    else:
        
        Xdf = get_glassnode_data(list_variables,Urls)
        Zdf = process_variables(Xdf)
        
        price = get_glassnode_price()

        Y = predcition_analysis(Zdf,price,'%Y-%m-%d',train_days)
        
        Y1 = Y.cumsum().reset_index()

        f = go.Figure()
        f.add_trace(go.Scatter(x=Y1['t'], y=Y1['Target'],
                            mode='lines',
                            name='Target'))
        
        f.add_trace(go.Scatter(x=Y1['t'], y=Y1['estimated'],
                            mode='lines',
                            name='Predicted'))

        
        f.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01))

        st.plotly_chart(f, use_container_width=True)
        prc = round(100*regression_cm(Y),2)
        st.write('#### Implied Precision:',f'{prc}%')
                  
except URLError as e:
    st.error(
        """
        **This demo requires internet access.**

        Connection error: %s
    """
        % e.reason
    )

# -


