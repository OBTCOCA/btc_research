#%%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import argrelextrema
from tqdm import tqdm
import mplfinance as mpf

import sys

sys.path.insert(0,'/Users/orentapiero/Documents/MyResearch/btc_research/')

from btc_functions.glassnode import *
from btc_functions.import_data import get_glassnode_price,get_glassnode_data
from btc_functions.variable_list_urls import *
from btc_functions.utilities import strided_app,strided_app2
from xgboost import XGBClassifier

plt.rcParams['figure.figsize'] = [18, 10]
sns.set()
#%%
Urls['transfers_volume_sum'] = 'https://api.glassnode.com/v1/metrics/transactions/transfers_volume_sum'
Urls['ssr'] = 'https://api.glassnode.com/v1/metrics/indicators/ssr'
#%%
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
            'marketcap_thermocap_ratio',
            'difficulty_latest', 
            'non_zero_count']

features = get_glassnode_data(selected,Urls)
features = features.loc[:'2019']
#%%
px = np.log(ohlc.Close)
target_df = pd.DataFrame(px)
L = 5

target_df['R'] = px.diff()
target_df['RV'] = np.sqrt((target_df['R']**2).rolling(L).sum())
target_df['FR'] = (px.shift(-L) - px)
target_df['FR2RV'] = target_df['FR']/target_df['RV']
target_df['Target'] = 0.
target_df.loc[target_df.FR2RV>=1.,'Target']=1.
target_df.loc[target_df.FR2RV<=-1.,'Target']=-1.

target_df = target_df.dropna()
#%%
px = np.log(ohlc.Close)
target_df = pd.DataFrame(px)
L = 5

target_df['R'] = px.diff()
target_df['RV'] = np.sqrt((target_df['R']**2).rolling(L).sum())
target_df['FR'] = (px.shift(-L) - px)
target_df['FR2RV'] = target_df['FR']/target_df['RV']
target_df['Target'] = np.nan

target_df.loc[target_df.FR2RV>=1.,'Target']=1.
target_df.loc[target_df.FR2RV<=-1.,'Target']=2.
target_df.loc[:,'Target'].fillna(0,inplace = True)

target_df = target_df.dropna()
# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

def on_chain_classifier(data,train_period,cv_period,
                        freqency = '%Y-%m-%d'):
    
    train_data = data.loc[data.index.strftime(freqency).isin(train_period)]
    
    try:
        cv_data = data.loc[data.index.strftime(freqency).isin(cv_period)]
    except:
        cv_data = data.loc[data.index.strftime(freqency).isin([cv_period])]
        
    y_train,X_train = train_data['Target'],train_data.loc[:,train_data.columns != 'Target']
    y_cv,X_cv = cv_data['Target'],cv_data.loc[:,cv_data.columns != 'Target']
    
    clf = DecisionTreeClassifier()

    clf.fit(X_train,y_train)
    yhat_cv = pd.Series(clf.predict(X_cv),index = y_cv.index,name = 'y_hat_cv')
    res = pd.concat([yhat_cv,y_cv],axis=1)
    return res
# %%

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
            'marketcap_thermocap_ratio',
            'non_zero_count']
#             'difficulty_latest', 


feats = features[selected]# .rolling(1).mean()

data = pd.concat([target_df['Target'],feats],axis=1).dropna()

dates = np.unique(data.index.strftime('%Y-%m-%d'))
strided_dates = strided_app(dates,252,L)
M = strided_dates.shape[0]

results = [
    on_chain_classifier(data,
                        strided_dates[t,:-L],
                        strided_dates[t,-L:],
                        freqency = '%Y-%m-%d')
    
    for t in tqdm(range(M))]

results = pd.concat(results,axis=0).sort_index()
print('accuracy:',accuracy_score(results.Target,results.y_hat_cv))
# %%
