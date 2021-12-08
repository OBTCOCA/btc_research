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
#%%
GLASSNODE_API_KEY = '1vUcyF35hTk9awbNGszF0KcLuYH'

self = GlassnodeClient()
self.set_api_key(GLASSNODE_API_KEY)

url = 'https://api.glassnode.com/v1/metrics/entities/min_1k_count'

a ='BTC'
c = 'native'
i='24h'

data = self.get(url,a,i,c)

# %%
