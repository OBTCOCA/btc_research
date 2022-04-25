# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import time

sys.path.insert(0,'/Users/orentapiero/Documents/MyResearch/on_chain_project/')

from btc_functions.glassnode import *
from btc_functions.import_data import get_glassnode_price,get_glassnode_data
from btc_functions.variable_list_urls import *
from btc_functions.utilities import strided_app,strided_app2

from xgboost import XGBClassifier

plt.rcParams['figure.figsize'] = [18, 10]
sns.set()
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
            'marketcap_thermocap_ratio',
            'non_zero_count']

features = get_glassnode_data(selected,Urls)
features = features.loc[:'2019']

#%%

data = pd.concat([ohlc.Close,features],axis = 1).dropna()

# %%
L = 5
th = 1

strided_dates = strided_app(data.index.values,252,1)

df_train = data.loc[strided_dates[0,:-1]].copy()
df_cv = data.loc[strided_dates[0,-1]].copy()

px = np.log(df_train.Close)

Fr = 100*(px.shift(-L)-px)
target = pd.Series(0,index = Fr.index,name = 'target')
target[Fr>=th] = 1
target[Fr<=-th] = 2
target[Fr.isna()] = np.nan

train_set = pd.concat([target,df_train.iloc[:,1:]],axis = 1).dropna()
y_train,X_train =  train_set['target'],train_set.loc[:,train_set.columns != 'target']

clf = XGBClassifier(n_estimators=1,
                     max_depth=5,
                     max_leaves=64,
                     eta=0.1,
                     reg_lambda=0,
                     tree_method='hist',
                     eval_metric='logloss',
                     use_label_encoder=False,
                     random_state=1000,
                     n_jobs=-1)


start = time.time()
clf.fit(X_train.values,y_train.values)
elapsed = time.time() - start
print(f'XGB Training ran in {elapsed:.5f} seconds')

# %%
