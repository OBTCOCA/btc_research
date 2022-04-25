# %%
# %%%
import pandas as pd
import numpy as np 
import math
import sys
sys.path.insert(0, '/Users/orentapiero/MyResearch') 


from FILTERS.utilities import strided_app
# %%

def recurrsion(x,pars,lag):
    T = len(x)
    bp = np.zeros_like(x)

    for t in range(lag,T):
        bp[t] = pars[0]*(x[t]-x[t-lag])+pars[1]*bp[t-1]+pars[2]*bp[t-2] 
    return bp

def zipper_recurrsion(x,pars,lag):
    T = len(x)
    bp = np.zeros_like(x)
    
    for t in range(-(T-lag),T):
        bp[t] = pars[0]*(x[t]-x[t-lag])+pars[1]*bp[t-1]+pars[2]*bp[t-2] 
        
#     for t in range(1,T):
#         bp[t] = pars[0]*(x[t]-x[t-lag])+pars[1]*bp[t-1]+pars[2]*bp[t-2] 


    return bp


def BandPass(series,period,delta,lag):
    beta = math.cos(2*math.pi/period)
    gamma = 1/math.cos(4*math.pi*delta/period)
    alpha = gamma-math.sqrt(gamma*gamma-1)

    pars = (0.5*(1-alpha),beta*(1+alpha),-alpha)
    bp = recurrsion(series.values,pars,lag)

    return pd.Series(bp,index = series.index)
# %%
# def TruncatedBandPass(series,period,delta,Ltrunc,lag):
#     beta = math.cos(2*math.pi/period)
#     gamma = 1/math.cos(4*math.pi*delta/period)
#     alpha = gamma-math.sqrt(gamma*gamma-1)

#     pars = (0.5*(1-alpha),beta*(1+alpha),-alpha)

#     x = series.values
#     bp = np.empty_like(x)
#     bp[:] = np.nan
    
#     x_strided = strided_app(x,Ltrunc,1)
#     lst = [recurrsion(x_strided[j,:],pars,lag)[-1] for j in range(x_strided.shape[0])]
#     bp[(Ltrunc-1):] = lst
#     return pd.Series(bp,index = series.index)

def TruncatedBandPass(series,period,delta,Ltrunc):
    z = series.values

    x_strided = strided_app(z,Ltrunc,1).T.copy()

    beta = math.cos(2*math.pi/period)
    gamma = 1/math.cos(4*math.pi*delta/period)
    alpha = gamma-math.sqrt(gamma*gamma-1)

    pars = (0.5*(1-alpha),beta*(1+alpha),-alpha)

    T = x_strided.shape[0]
    bp = np.zeros_like(x_strided)
    BP = np.zeros_like(z)
    BP[:] = np.nan

    for t in range(2,T):
        bp[t,:] =pars[0]*(x_strided[t,:]-x_strided[t-2,:])+pars[1]*bp[t-1,:]+pars[2]*bp[t-2,:] 

    BP[(Ltrunc-1):] = bp[-1,:]
    BP = pd.Series(BP,index=series.index)
    return BP
# %%
# def TruncatedZipperBandPass(series,period,delta,Ltrunc,lag):
#     beta = math.cos(2*math.pi/period)
#     gamma = 1/math.cos(4*math.pi*delta/period)
#     alpha = gamma-math.sqrt(gamma*gamma-1)

#     pars = (0.5*(1-alpha),beta*(1+alpha),-alpha)

#     x = series.values
#     bp = np.empty_like(x)
#     bp[:] = np.nan

#     x_strided = strided_app(x,Ltrunc,1)
#     lst = [zipper_recurrsion(x_strided[j,:],pars,lag)[-1] for j in range(x_strided.shape[0])]
#     bp[(Ltrunc-1):] = lst
#     return pd.Series(bp,index = series.index)

def TruncatedZipperBandPass(series,period,delta,Ltrunc):
    z = series.values

    x_strided = strided_app(z,Ltrunc,1).T.copy()

    beta = math.cos(2*math.pi/period)
    gamma = 1/math.cos(4*math.pi*delta/period)
    alpha = gamma-math.sqrt(gamma*gamma-1)

    pars = (0.5*(1-alpha),beta*(1+alpha),-alpha)

    T = x_strided.shape[0]
    bp = np.zeros_like(x_strided)
    BP = np.zeros_like(z)
    BP[:] = np.nan

    for t in range(-(T-2),T):
        bp[t,:] =pars[0]*(x_strided[t,:]-x_strided[t-2,:])+pars[1]*bp[t-1,:]+pars[2]*bp[t-2,:] 

    BP[(Ltrunc-1):] = bp[-1,:]
    BP = pd.Series(BP,index=series.index)
    return BP
