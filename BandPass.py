import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from numba import njit,jit
from numba.typed import List
from tqdm import tqdm 



class Detrend(object):
    
    def __init__(self,theta):
        self.theta = theta
    
    def detrend(self,series):
        x = series.values.tolist()
        T = len(x)
        z = T*[0.0]
        
        typed_x = List()
        typed_z = List()

        [typed_x.append(j) for j in x]
        [typed_z.append(j) for j in z]
     
        z = self._detrend(self.theta,typed_x,typed_z,T)
        return pd.Series(z,index = series.index)
    
    @staticmethod
    @njit 
    def _detrend(theta,x,z,T):
        for t in range(1,T):
            z[t] = (x[t]-x[t-1])+theta*z[t-1]
        return z

class BandPass(object):
    
    
    def __init__(self,period,delta):
        self.period = period 
        self.delta = delta

    def filter(self,series):
        beta = math.cos(2*math.pi/self.period)
        gamma = 1/math.cos(4*math.pi*self.delta/self.period)
        alpha = gamma-math.sqrt(gamma*gamma-1)

        pars = (0.5*(1-alpha),beta*(1+alpha),-alpha)

        x = series.values.tolist()
        T = len(x)
        bp = T*[0.0]
        
        typed_x = List()
        typed_bp = List()
        
        [typed_x.append(j) for j in x]
        [typed_bp.append(j) for j in bp]
        
        bp = self._recurssion(pars,typed_x,typed_bp,T)
        return pd.Series(bp,index = series.index)

    @staticmethod
    @njit 
    def _recurssion(pars,x,bp,T):

        for t in range(1,T):
            if t == 1:
                bp[t] = pars[0]*10*(x[t+1]-x[t+72])
            else:
                bp[t] = pars[0]*(x[t]-x[t-2])+pars[1]*bp[t-1]+pars[2]*bp[t-2] 
        return bp
