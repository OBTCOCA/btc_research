import numpy as np
import pandas as pd

def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

def WMA(Rate,L):
    x = Rate.values
    
    if len(x.shape) > 1:
        x = x.reshape(1,-1)[0]
    
    wma = np.empty_like(x)
    wma[:] = np.nan

    x_strided = strided_app(x,L,1)

    K = L*(L+1)/2
    weights = np.arange(1,(L+1))/K

    wma[(L-1):] = np.dot(x_strided,weights)
    out = pd.Series(wma,index = Rate.index, name = 'WMA_'+str(L))
    return out
