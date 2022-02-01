# %%
import pandas as pd
import numpy as np
# %%
class POT(object):

    def __init__(self,High,Low):
        self.High = High
        self.Low = Low
        
        
    def abs(self,L):
        High = self.High
        Low = self.Low
        
        L = int(L)
        
        r_high = 10000*np.log(High).diff().fillna(0)
        r_low  = 10000*np.log(Low).diff().fillna(0)

        r_high_sign = np.sign(r_high)
        r_low_sign  = np.sign(r_low)
        
        r_high_sign.loc[r_high_sign < 0] = 0
        r_low_sign.loc[r_low_sign > 0]   = 0

        rolling_high = r_high_sign.rolling(L).sum()
        rolling_low = r_low_sign.abs().rolling(L).sum()

        pot = rolling_high/(rolling_low+rolling_high)
        return pot
    
    def bps(self,L):
        High = self.High
        Low = self.Low
        
        L = int(L)
        
        r_high = 10000*np.log(High).diff().fillna(0)
        r_low  = 10000*np.log(Low).diff().fillna(0)

        r_high_sign = np.sign(r_high)
        r_low_sign  = np.sign(r_low)
        
        r_high_sign.loc[r_high_sign < 0] = 0
        r_low_sign.loc[r_low_sign > 0]   = 0

        bps_high = r_high_sign*r_high
        bps_low  = r_low_sign*r_low

        rolling_bps_high = bps_high.rolling(L).sum()
        rolling_bps_low = bps_low.rolling(L).sum()

        pot_bps = rolling_bps_high/(rolling_bps_high+rolling_bps_low)
        
        return pot_bps
        
        

