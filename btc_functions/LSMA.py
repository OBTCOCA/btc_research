import numpy as np
import pandas as pd

def ols_fcast(y,order = 1):
    n = len(y)
    x = np.arange(n)
    res = np.polyfit(x, y, order)
    ycast = res[0]*(n)+res[1]
    
    if order == 1:
        ycast = res[0]*(n)+res[1]
    elif order == 2:
        ycast = res[0]*(n**2)+res[1]*(n)+res[2]
    elif order == 3:
        ycast = res[0]*(n**3)+res[1]*(n**2)+res[2]*(n) +res[3]
    elif order == 4:
        ycast = res[0]*(n**4)+res[1]*(n**3)+res[2]*(n**2) +res[3]*(n)+res[4]
        
    return ycast

def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

class LSMA(object):
    def __init__(self,S):
        self.S = S
        
    def predict(self,window,order):
        S = self.S
        
        S_strided = strided_app(S.values,window,1)
        lsma = np.empty_like(S.values)
        lsma[:] = np.nan

        lsma[(window-1):] = np.array([ols_fcast(S_strided[i,:],order) for i in range(S_strided.shape[0])])
        lsma = pd.Series(lsma,index = S.index,name = 'lsma_'+str(window))
        return lsma

