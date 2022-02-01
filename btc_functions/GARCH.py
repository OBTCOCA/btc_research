# +
import numpy as np
import pandas as pd

from scipy.optimize import fmin_slsqp

# -

def gjr_garch_likelihood(pars,data,sigma2 = None,out = None):
    ''' ARMA-GARCH model '''
    mu,rho,theta,omega,alpha,gamma,beta = pars
    
    T = np.size(data,0)
    
    eps = np.zeros_like(data)
    sigma2 = np.zeros_like(data)
    
    if sigma2 is None:
        sigma2 = np.zeros_like(data)
        sigma2[0] = np.var(data[:5])

    for t in range(1,T):
        eps[t] = data[t] - mu - rho*data[t-1] - theta*eps[t-1]
        sigma2[t] = omega + alpha*eps[t-1]*eps[t-1] + gamma*eps[t-1]*eps[t-1]*(eps[t-1]<0) + beta*sigma2[t-1]
    
    logliks = 0.5*(np.log(2*np.pi) + np.log(sigma2) + eps**2/sigma2)
    loglik = np.sum(logliks)
    
    if out is None:
        return loglik
    else:
        return loglik, logliks, copy(sigma2)


def gjr_constraint(pars, data, sigma2=None, out=None):
    ''' Constraint that alpha+gamma/2+beta<=1'''
    mu,rho,theta,omega,alpha,gamma,beta = pars
    return np.array([1-alpha-gamma/2-beta])


