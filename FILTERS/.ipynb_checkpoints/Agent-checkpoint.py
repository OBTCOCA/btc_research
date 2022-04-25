# %%
import sys 

if sys.platform == 'linux':
    sys.path.insert(1,'/home/oren/Research/')
elif sys.platform == 'darwin':
    sys.path.insert(1,'/Users/orentapiero/Research/')
else:
    sys.path.insert(1,'C:/Research/')

from models.BaseAgent import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import statsmodels.api as sm 

from tqdm import tqdm 


import warnings
warnings.filterwarnings('ignore')


# %%
def entry(self,x):
                
    BP = x['entry_indicator']
    Cond = x['entry_condition']
    
    dsign_BP = BP.groupby(x.index.date).apply(lambda x: np.sign(x).diff())
    
    if (self.pos == 1):
        mask = (dsign_BP == 2) & (Cond == True)
            
    elif (self.pos == -1):
        mask = (dsign_BP == -2) & (Cond == True)

    self.a_entry = mask
    return mask

def exit(self,x,a_entries=None):
    a_exits1 = self.exit1(x,a_entries)
    a_exits2 = self.exit2(x,a_entries)
    
    a_exits = pd.concat([a_exits1,a_exits2],1)        
    a_exits = a_exits.min(1)
    
    return a_exits

class Agent(BaseAgent):
    
    def __init__(self,
                 # base parameters
                 fwd=1,
                 pos=1,
                 max_chain=1, 
                 max_spread=2,
                 sl=None, ## No SL 
                 tp=None, ## No TP 
                 first_entry_hour = '7:00',
                 last_entry_hour = '19:00',
                 fri_last_entry_hour = '18:00',
                 last_exit_hour = '19:05:00', 
                 friday_last_exit_hour = '19:05:00',
                 data_lookback = None,
                 start = None,
                 cv_start = None,
                 end = None,
                 optimizer = None,
                 verbose = False,
                 bars = None,
                 exit = exit,
                 entry = entry):
        
        super().__init__(fwd = fwd,
                         pos = pos,
                         max_chain = max_chain,
                         max_spread = max_spread,
                         sl = sl,
                         tp = tp,
                         first_entry_hour = first_entry_hour,
                         last_entry_hour = last_entry_hour,
                         fri_last_entry_hour = fri_last_entry_hour,
                         last_exit_hour = last_exit_hour, 
                         friday_last_exit_hour = friday_last_exit_hour,
                         data_lookback = data_lookback,
                         start = start,
                         cv_start = cv_start,
                         end = end,
                         verbose = verbose,
                         optimizer = optimizer,
                         bars = bars,
                         exit = exit,
                         entry = entry)
                
        
    def exit1(self,x,a_entries = None):
        BP = x['exit_indicator']

        dsign_BP = BP.groupby(x.index.date).apply(lambda x: np.sign(x).diff())

        if (self.pos == 1):
            mask = (dsign_BP == -2) 

        elif (self.pos == -1):
            mask = (dsign_BP == 2) 
            
        a_exits = pd.Series(index=a_entries.index,dtype='datetime64[ns]')
        M_index = self._get_entries_strided_matrix(x.index,a_entries)
        exits = self._get_model_exits(M_index,mask, a_entries)
        a_exits.loc[a_entries] = exits
        self.a_exit = mask
        self.a_exits = a_exits
        self.M_index = M_index
        
        return a_exits        
    
    
    def exit2(self,x,a_entries = None):
        
        M_pnl = self._get_pnl_matrix(x,a_entries)  
        M_pnl = pd.DataFrame(M_pnl.T, columns = a_entries[a_entries].index)
        self.M_pnl = M_pnl
        
        entries = M_pnl.columns 
        RollSL = (M_pnl - self.sl).cummax()
        RollSL = RollSL[M_pnl<RollSL]
        
        out = []
        for j in entries:
            tr = RollSL.loc[:,j].dropna()
            if len(tr)>0:
                n = tr.index[0]*5
                exit_ = j + pd.Timedelta(str(n)+'T')
                out.append({'en':j,'ex':exit_})
        
        exits = pd.DataFrame(out)
        exits.index = exits.en
        exits = exits['ex']
        
        
        #exits = pd.to_datetime(exits).values ## <---- 
        a_exits = pd.Series(index=x.index, dtype = 'datetime64[ns]')
        a_exits.loc[exits.index] = exits
        return a_exits
        
    

# %%
