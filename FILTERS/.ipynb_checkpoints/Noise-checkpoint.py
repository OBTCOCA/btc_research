# +
import sys 
import numpy as np
import pandas as pd

from tqdm import tqdm
# -

sys.path.insert(0, '/Users/orentapiero/MyResearch') 


from FILTERS.utilities import strided_app

from scipy import stats

def reg(x,y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    err = y - intercept-slope*x
    return err.std()

def reg_pos(x,y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    err = y - intercept-slope*x
    return err[err<0].mean()

def reg_neg(x,y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    err = y - intercept-slope*x
    return err[err<0].mean()

def noise_estimate(x,L,verbose = False):
    z = x.values
    idx = x.index
    
    z_strided = strided_app(z,L,1)
    sigma = np.empty_like(z)
    sigma[:] = np.nan
    trend = np.arange(L) + 1
    
    if verbose == False:
        sigma[(L-1):] = np.array([reg(trend,z_strided[j,:]) for j in range(z_strided.shape[0])])
    else:
        sigma[(L-1):] = np.array([reg(trend,z_strided[j,:]) for j in tqdm(range(z_strided.shape[0]))])
        
    Sigma = pd.Series(sigma,index = idx)
    return Sigma

def noise_estimate_pos(x,L,verbose = False):
    z = x.values
    idx = x.index
    
    z_strided = strided_app(z,L,1)
    sigma = np.empty_like(z)
    sigma[:] = np.nan
    trend = np.arange(L) + 1
    
    if verbose == False:
        sigma[(L-1):] = np.array([reg_pos(trend,z_strided[j,:]) for j in range(z_strided.shape[0])])
    else:
        sigma[(L-1):] = np.array([reg_pos(trend,z_strided[j,:]) for j in tqdm(range(z_strided.shape[0]))])
        
    Sigma = pd.Series(sigma,index = idx)
    return Sigma

def noise_estimate_neg(x,L,verbose = False):
    z = x.values
    idx = x.index
    
    z_strided = strided_app(z,L,1)
    sigma = np.empty_like(z)
    sigma[:] = np.nan
    trend = np.arange(L) + 1
    
    if verbose == False:
        sigma[(L-1):] = np.array([reg_neg(trend,z_strided[j,:]) for j in range(z_strided.shape[0])])
    else:
        sigma[(L-1):] = np.array([reg_neg(trend,z_strided[j,:]) for j in tqdm(range(z_strided.shape[0]))])
        
    Sigma = pd.Series(sigma,index = idx)
    return Sigma
