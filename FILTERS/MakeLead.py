# %%
import pandas as pd 
import numpy as np 
# %%
def Lead(x,alpha1,lag = 1):
    lead = np.zeros_like(x.values)
    lead[0] = x.values[0]
    
    T = len(x)
    
    for t in range(lag,T):
        lead[t] = 2*x.values[t]+(alpha1-2)*x.values[t-lag]+(1-alpha1)*lead[t-1]
    
    lead = pd.Series(lead,index=x.index)
    return lead   
