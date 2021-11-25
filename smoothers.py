import pandas as pd
import numpy as np 

def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

def VWAP(Price,Volume,L):
    x = Price.values
    v = Volume.values
    t = Price.index

    x_strided,v_strided = strided_app(x,L,1),strided_app(v,L,1) 
    vwap = np.empty_like(x)
    vwap[:] = np.nan

    vwap[(L-1):] = [(x_strided[i]*v_strided[i]).sum()/v_strided[i].sum() for i in range(x_strided.shape[0])]

    return pd.Series(vwap,index=t)

def MA(Price,L):
    return Price.rolling(L).mean()

def EMA(Price,L):
    return Price.ewm(span = L).mean()

