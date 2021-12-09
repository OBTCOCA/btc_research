#%%
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


st.set_page_config(
     page_title="DIY AI On-Chain BTC Predictions",
     page_icon="🤖",
     layout="wide",
     initial_sidebar_state="auto",
     menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        #'About': "# This is a header. This is an *extremely* cool app!"
     }
 )

#%%
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

Indicators = ['rhodl_ratio',
              'cvdd',
              'balanced_price_usd',
              'hash_ribbon',
              'difficulty_ribbon',
              'difficulty_ribbon_compression',
              'nvt',
              'nvts',
              'velocity',
              'nvt_entity_adjusted',
              'cdd_supply_adjusted',
              'cdd_supply_adjusted_binary',
              'average_dormancy_supply_adjusted',
              'spent_output_price_distribution_ath',
              'spent_output_price_distribution_percent',
              'puell_multiple',
              'sopr_adjusted',
              'reserve_risk',
              'sopr_less_155',
              'sopr_more_155',
              'hodler_net_position_change',
              'hodled_lost_coins',
              'cyd',
              'cyd_supply_adjusted',
              'cyd_account_based',
              'cyd_account_based_supply_adjusted',
              'cdd90_age_adjusted',
              'cdd90_account_based_age_adjusted',
              'sopr',
              'cdd',
              'asol',
              'msol',
              'average_dormancy',
              'liveliness',
              'unrealized_profit',
              'unrealized_loss',
              'net_unrealized_profit_loss',
              'nupl_less_155',
              'nupl_more_155',
              'sopr_account_based',
              'cdd_account_based',
              'asol_account_based',
              'msol_account_based',
              'dormancy_account_based',
              'dormancy_flow',
              'liveliness_account_based',
              'mvrv_account_based',
              'rcap_account_based',
              'unrealized_profit_account_based'
              'unrealized_loss_account_based',
              'net_unrealized_profit_loss_account_based',
              'nupl_less_155_account_based',
              'nupl_more_155_account_based',
              'net_realized_profit_loss',
              'realized_profit_loss_ratio',
              'stock_to_flow_ratio',
              'stock_to_flow_deflection',
              'realized_profit',
              'realized_loss',
              'ssr',
              'ssr_oscillator', 
              'utxo_realized_price_distribution_ath',
              'utxo_realized_price_distribution_percent',
              'soab',
              'sol_1h',
              'sol_1h_24h',
              'sol_1d_1w',
              'sol_1w_1m',
              'sol_1m_3m',
              'sol_3m_6m', 
              'sol_6m_12m',
              'sol_1y_2y',
              'sol_2y_3y',
              'sol_3y_5y',
              'sol_5y_7y',
              'sol_7y_10y',
              'sol_more_10y']

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

Derivatives = ['futures_funding_rate_perpetual',
               'futures_funding_rate_perpetual_all',
               'futures_open_interest_cash_margin_sum',
               'futures_open_interest_crypto_margin_sum',
               'futures_open_interest_crypto_margin_relative',
               'futures_estimated_leverage_ratio',
               'futures_volume_daily_sum',
               'futures_volume_daily_perpetual_sum',
               'futures_open_interest_sum',
               'futures_open_interest_perpetual_sum',
               'futures_liquidated_volume_short_sum',
               'futures_liquidated_volume_short_mean',
               'futures_liquidated_volume_long_sum',
               'futures_liquidated_volume_long_mean',
               'futures_liquidated_volume_long_relative',
               'futures_volume_daily_sum_all',
               'futures_volume_daily_perpetual_sum_all',
               'futures_open_interest_sum_all',
               'futures_open_interest_perpetual_sum_all',
               'options_volume_daily_sum',
               'options_open_interest_sum',
               'options_open_interest_distribution',
               'futures_open_interest_latest',
               'futures_volume_daily_latest']

Institutions = ['grayscale_holdings_sum',
                'grayscale_flows_sum',
                'grayscale_premium_percent',
                'grayscale_aum_sum',
                'grayscale_market_price_usd',
                'purpose_etf_holdings_sum',
                'purpose_etf_flows_sum',
                'qbtc_holdings_sum',
                'qbtc_flows_sum',
                'qbtc_premium_percent',
                'qbtc_aum_sum',
                'qbtc_market_price_usd']

urls = []

for a in Addresses:
    urls += [URLS['Addresses']+a]

for b in Blockchain:
    urls += [URLS['Blockchain']+b]

for d in Distribution:
    urls += [URLS['Distribution']+d]

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

for m2 in Derivatives:
    urls += [URLS['Derivatives']+m2]

for m3 in Institutions:
    urls += [URLS['Institutions']+m3]    

Urls = dict()

for u in urls:
    Urls[u.split('/')[-1]] = u


#%%
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

    for u in stqdm(urls):
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
        pr = pd.concat([train_price,cv_price],axis = 0).sort_index().fillna(method = 'ffill')
        
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

    if Xdf.shape[1] == 1:
        tomorrow = self.model.predict(Xdf.iloc[-1].values.reshape(-1,1))[0]
    else:
        tomorrow = self.model.predict(Xdf.iloc[-1].values.reshape(1,-1))[0]

    return Y,tomorrow


# +
# price = get_glassnode_price()
# list_variables = ['sopr','mvrv_z_score','deltacap_usd']
# Xdf = get_glassnode_data(list_variables,Urls)
# Zdf = process_variables(Xdf)
# Y = predcition_analysis(Zdf,price,'%Y-%m-%d',120)

# +
#%%
try:
    list_variables = st.multiselect("Choose on chain indicators", list(Urls.keys()), ["nvt", "msol"])
    train_days = st.slider('Number of days to train?', 10,252)
    chart_start_year = st.slider('Start year for plotting results?', 2014,2021)

    if not list_variables:
        st.error("Please select at least one indicators.")
    else:
        
        if st.button('Train Model'):
            st.write('Retreiving data and processing variables')
            try:
                Xdf = get_glassnode_data(list_variables,Urls)
                Zdf = process_variables(Xdf)
                st.success('Done !')
            except:
                st.exception('Failed')

            st.write('')
            st.write('Retrieving prices')
            try:
                price = get_glassnode_price()
                st.success('Done !')
            except:
                st.exception('Failed')

            st.write('')
            st.write('Starting prediction analysis')
            Y,tomorrow = predcition_analysis(Zdf,price,'%Y-%m-%d',train_days)
            
            R = np.exp(Y.loc[str(chart_start_year):]/100)
            init_price = price.shift(1).loc[R.index]
            
            Y2 = pd.concat([R.iloc[:,0]*(init_price).rename('Target'),
                            R.iloc[:,1]*(init_price).rename('Estimated')],
                            axis=1).reset_index()

            Y2.columns = ['t','Target','Estimated']

            f = go.Figure()
            f.add_trace(go.Scatter(x=Y2['t'], y=Y2['Target'],
                                mode='lines',
                                name='Target'))
            
            f.add_trace(go.Scatter(x=Y2['t'], y=Y2['Estimated'],
                                mode='lines',
                                name='Predicted'))
            f.update_yaxes(type="log")

            f.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01))
            #st.metric(label='Accuracy', value=f'{prc}%')
            
            st.plotly_chart(f, use_container_width=True)
            prc = round(100*regression_cm(Y),2)
            st.write('#### Implied Precision:',f'{prc}%')
            st.write('')
            pred =  round(tomorrow,2)
            st.write('#### Prediction for tomorrow',f'{pred}%')
            st.write('#### Price prediction for tomorrow',f'{price.iloc[-1]*(1+pred/100)}%')
            st.write('')
            #col1, col2, col3 = st.columns(3)
            #with col1:
            #st.metric(label='Accuracy', value=f'{prc}%')
            st.write('')
            #with col2:
            #st.metric(label="Today's Change Prediction", value=f'{pred}%')
            st.write('')
            #with col3:
            #st.write(st.metric(label="Today's Price Prediction", value=str(f'{price.iloc[-1]*(1+pred/100)}%')))
            st.write('')
            st.write('#### Forecast evaluation by regressing "Target" on "Estimated"')
            X = sm.add_constant(Y['estimated'])
            y = Y['Target']
            mod = sm.OLS(y,X).fit()
            st.write(mod.summary())
            st.write('')
            fig = px.scatter(x=Y['estimated'], y=Y['Target'],trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
            st.write('X - axis: estimated')
            st.write('Y - Realized')

except URLError as e:
    st.error(
        """
        **This demo requires internet access.**

        Connection error: %s
    """
        % e.reason
    )

# -


