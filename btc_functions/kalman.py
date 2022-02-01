# %%
import numpy as np
import pandas as pd
from numba import njit, prange,jit
from numba.typed import List

# %%
@njit
def kf(z, F, H, Q, R, x, P, N):
    for t in range(1, N):
        x[t] = F @ x[t - 1]
        P[t] = F @ P[t - 1] @ F.T + Q

        d = z[t] - H @ x[t]
        s = H @ P[t] @ H.T + R
        k = (P[t] @ H.T) / s
        x[t] = x[t] + k.T * d
        P[t] = P[t] - k @ H @ P[t]
    return x

def akf(alpha, z, F, H, Q, R, x, P, N):
    for t in range(1, N):
        x[t] = F @ x[t - 1]
        P[t] = F @ P[t - 1] @ F.T + Q[t - 1]

        d = z[t] - H @ x[t]
        s = H @ P[t] @ H.T + R[t - 1]
        k = (P[t] @ H.T) / s[0]
        x[t] = x[t] + k.T * d
        P[t] = P[t] - k @ H @ P[t]
        e = z[t] - H @ x[t]
        
        R[t] = alpha * R[t - 1] + (1-alpha) * (e ** 2 + H @ P[t] @ H.T)
        Q[t] = alpha * Q[t - 1] + (1-alpha) * ((d ** 2) * (k @ k.T))
        
    return x


class kfilter(object):
    
    def __init__(self,F=None, H=None, c=None, Q=None, R=None,P0=None,x0 = None):
        N_states = F.shape[0]
        N_obs = H.ndim

        self.F = F
        self.H = H.reshape(N_obs, N_states)
        self.c = np.zeros(N_states) if c is None else c
        self.Q = np.eye(N_states) if Q is None else Q
        self.R = np.eye(N_obs) if R is None else R.reshape(N_obs, N_obs)
        self.x0 = np.zeros(N_states) if x0 is None else x0
        self.P0 = np.ones([N_states, N_states]) if P0 is None else P0
        
    def filter(self,series):
        
        z = series.values
        N = len(z)

        x = np.zeros([N,3])
        P = np.zeros([N,3,3])
        
        x[0,:] = self.x0
        P[0] = self.P0
            
        x = kf(z, self.F, self.H, self.Q, self.R, x, P, N)
                        
        return pd.DataFrame(x,index = series.index) 
            
            
            
    def afilter(self,series,alpha):
        z = series.values
        
        z = series.values
        N = len(z)

        x = np.zeros([N,3])
        P = np.zeros([N,3,3])
        Q = np.zeros([N,3,3])
        Q[0] = self.Q
        
        R = [0]*N
        R[0] = self.R[0]
        
        x[0,:] = self.x0
        P[0] = self.P0
        
         
        x = akf(alpha,z, self.F, self.H, Q, R, x, P, N)
        
        return pd.DataFrame(x,index = series.index)


