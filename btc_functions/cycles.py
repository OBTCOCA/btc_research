# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: 'Python 3.9.7 64-bit (''base'': conda)'
#     language: python
#     name: python397jvsc74a57bd0c6e4e9f98eb68ad3b7c296f83d20e6de614cb42e90992a65aa266555a3137d0d
# ---

import pandas as pd
import numpy as np


# +
def simple_cycle(series,alpha):
    x = series.values
    
    A = (1-0.5*alpha)**2
    B = 2*(1-alpha)
    C = (1-alpha)**2
    
    c = np.zeros_like(x)
    T = len(x)
    
    for t in range(2,T):
        if t < 7:
            c[t] = (x[t]-2*x[t-1]+x[t-2])/4
        else:
            c[t] = A*(x[t]-2*x[t-1]+x[t-2])+B*c[t-1]-C*c[t-2]
    return pd.Series(c,index = series.index)


def simple_quadrature(series):
    x = series.values
    q = np.zeros_like(x)
    T = len(x)
    
    for t in range(6,T):
        q[t] = 0.0962*x[t]+0.5769*x[t-2]-0.5769*x[t-4]-0.0962*x[t-6]
    return pd.Series(q,index=series.index)
# -


