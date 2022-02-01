# %%
import sys

sys.path.insert(0, '/Users/orentapiero/MyResearch') 

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import math
import statsmodels.api as sm
import scipy.stats

from tqdm.notebook import tqdm
from FILTERS.utilities import strided_app
from FILTERS.hpfilter import hprescott
# %%

def hp_trend_wrapper(x,side = 2,smooth = 1600):
    cycle, trend = hprescott(x, side = side, smooth = smooth)
    return trend[-1]

def roll_hp_trend(series,side = 2,L = 12,smooth = 1600,verbose = False):
    z = series.values
    t = series.index

    z_strided = strided_app(z,L,1)

    smth = np.empty_like(z)
    smth[:] = np.nan
    
    if verbose is True:
        smth[(L-1):] = [hp_trend_wrapper(z_strided[i,:],side = side,smooth=smooth) for i in tqdm(range(z_strided.shape[0]))]
    else:
        smth[(L-1):] = [hp_trend_wrapper(z_strided[i,:],side = side,smooth=smooth) for i in range(z_strided.shape[0])]
        
    smth = pd.Series(smth,index = t)
    return smth

# %%

def hp_cycle_wrapper(x,side = 2,smooth = 1600):
    cycle, trend = hprescott(x, side = side, smooth = smooth)
    return cycle[-1]

def pseudo_hp_cycle(series,side = 2,L = 12,smooth = 1600,verbose = False):
    z = series.values
    t = series.index

    z_strided = strided_app(z,L,1)

    smth = np.empty_like(z)
    smth[:] = np.nan
    
    if verbose is True:
        smth[(L-1):] = [hp_cycle_wrapper(z_strided[i,:],side = side,smooth=smooth) for i in tqdm(range(z_strided.shape[0]))]
    else:
        smth[(L-1):] = [hp_cycle_wrapper(z_strided[i,:],side = side,smooth=smooth) for i in range(z_strided.shape[0])]
        
    smth = pd.Series(smth,index = t)
    return smth

# %%
def hp_avg_bps_trend_wrapper(x,N,side = 2,smooth = 1600):
    cycle, trend = hprescott(x, side = side, smooth = smooth)
    r = 10000*np.diff(np.log(trend))
    return np.mean(r[-N:])

def hp_avg_trend_bps(series,side = 2,L = 12,N = 6,smooth = 1600,verbose = False):
    z = series.values
    t = series.index

    z_strided = strided_app(z,L,1)

    smth = np.empty_like(z)
    smth[:] = np.nan
    
    if verbose is True:
        smth[(L-1):] = [hp_avg_bps_trend_wrapper(z_strided[i,:],N,side = side,smooth=smooth) for i in tqdm(range(z_strided.shape[0]))]
    else:
        smth[(L-1):] = [hp_avg_bps_trend_wrapper(z_strided[i,:],N,side = side,smooth=smooth) for i in range(z_strided.shape[0])]
    smth = pd.Series(smth,index = t)
    
    return smth
# %%
def hp_avg_bps_cycle_wrapper(x,N,side = 2,smooth = 1600):
    cycle, trend = hprescott(x, side = side, smooth = smooth)
    cycle *= 10000
    return np.mean(cycle[-N:])

def hp_avg_cycle_bps(series,side = 2,L = 12,N = 6,smooth = 1600,verbose = False):
    z = series.values
    t = series.index

    z_strided = strided_app(z,L,1)

    smth = np.empty_like(z)
    smth[:] = np.nan
    
    if verbose is True:
        smth[(L-1):] = [hp_avg_bps_cycle_wrapper(z_strided[i,:],N,side = side,smooth=smooth) for i in tqdm(range(z_strided.shape[0]))]
    else:
        smth[(L-1):] = [hp_avg_bps_cycle_wrapper(z_strided[i,:],N,side = side,smooth=smooth) for i in range(z_strided.shape[0])]
    smth = pd.Series(smth,index = t)
    
    return smth
# %%
def hp_avg_abs_bps_trend_wrapper(x,N,side = 2,smooth = 1600):
    cycle, trend = hprescott(x, side = side, smooth = smooth)
    r = 10000*np.diff(np.log(trend))
    return np.mean(np.abs(r[-N:]))

def hp_avg_abs_trend_bps(series,side = 2,L = 12,N = 6,smooth = 1600,verbose = False):
    z = series.values
    t = series.index

    z_strided = strided_app(z,L,1)

    smth = np.empty_like(z)
    smth[:] = np.nan
    
    if verbose is True:
        smth[(L-1):] = [hp_avg_abs_bps_trend_wrapper(z_strided[i,:],N,side = side,smooth=smooth) for i in tqdm(range(z_strided.shape[0]))]
    else:
        smth[(L-1):] = [hp_avg_abs_bps_trend_wrapper(z_strided[i,:],N,side = side,smooth=smooth) for i in range(z_strided.shape[0])]
        
    smth = pd.Series(smth,index = t)
    
    return smth


# %%
def hp_avg_abs_bps_cycle_wrapper(x,N,side = 2,smooth = 1600):
    cycle, trend = hprescott(x, side = side, smooth = smooth)
    cycle *= 10000
    return np.mean(np.abs(cycle[-N:]))

def hp_avg_abs_cycle_bps(series,side = 2,L = 12,N = 6,smooth = 1600,verbose = False):
    z = series.values
    t = series.index

    z_strided = strided_app(z,L,1)

    smth = np.empty_like(z)
    smth[:] = np.nan
    
    if verbose is True:
        smth[(L-1):] = [hp_avg_abs_bps_cycle_wrapper(z_strided[i,:],N,side = side,smooth=smooth) for i in tqdm(range(z_strided.shape[0]))]
    else:
        smth[(L-1):] = [hp_avg_abs_bps_cycle_wrapper(z_strided[i,:],N,side = side,smooth=smooth) for i in range(z_strided.shape[0])]
        
    smth = pd.Series(smth,index = t)
    
    return smth


# %%
def hp_trend_vel_wrapper(x,side = 2,smooth = 1600):
    cycle, trend = hprescott(x, side = side, smooth = smooth)
    return 10000*np.log(trend[-1]/trend[-3])

def roll_hp_velocity(series,side = 2,L = 12,smooth = 1600,verbose = False):
    z = series.values
    t = series.index

    z_strided = strided_app(z,L,1)

    smth = np.empty_like(z)
    smth[:] = np.nan
    
    if verbose is True:
        smth[(L-1):] = [hp_trend_vel_wrapper(z_strided[i,:],side = side,smooth=smooth) for i in tqdm(range(z_strided.shape[0]))]
    else:
        smth[(L-1):] = [hp_trend_vel_wrapper(z_strided[i,:],side = side,smooth=smooth) for i in range(z_strided.shape[0])]
        
    smth = pd.Series(smth,index = t)
    return smth
