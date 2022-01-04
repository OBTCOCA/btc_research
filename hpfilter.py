import numpy as np
from scipy.linalg import cho_factor, cho_solve
from statsmodels.tools.validation import array_like, PandasWrapper

def hprescott(X, side=2, smooth=1600, freq=''):
    '''
    Hodrick-Prescott filter with the option to use either the standard two-sided 
    or one-sided implementation. The two-sided implementation leads to equivalent
    results as when using the statsmodel.tsa hpfilter function
    
    Parameters
    ----------
    X : array-like
        The time series to filter (1-d), need to add multivariate functionality.
        
    side : int
           The implementation requested. The function will default to the standard
           two-sided implementation.
           
    smooth : float 
            The Hodrick-Prescott smoothing parameter. A value of 1600 is
            suggested for quarterly data. Ravn and Uhlig suggest using a value
            of 6.25 (1600/4**4) for annual data and 129600 (1600*3**4) for monthly
            data. The function will default to using the quarterly parameter (1600).

    freq : str
           Optional parameter to specify the frequency of the data. Will override
           the smoothing parameter and implement using the suggested value from
           Ravn and Uhlig. Accepts annual (a), quarterly (q), or monthly (m)
           frequencies.

    Returns
    -------
    
    cycle : ndarray
            The estimated cycle in the data given side implementation and the 
            smoothing parameter.
            
    trend : ndarray
            The estimated trend in the data given side implementation and the 
            smoothing parameter.
    
    References
    ----------
    Hodrick, R.J, and E. C. Prescott. 1980. "Postwar U.S. Business Cycles: An
        Empirical Investigation." `Carnegie Mellon University discussion
        paper no. 451`.
        
    Meyer-Gohde, A. 2010. "Matlab code for one-sided HP-filters."
        `Quantitative Macroeconomics & Real Business Cycles, QM&RBC Codes 181`.
    
    Ravn, M.O and H. Uhlig. 2002. "Notes On Adjusted the Hodrick-Prescott
        Filter for the Frequency of Observations." `The Review of Economics and
        Statistics`, 84(2), 371-80.
    
    Examples
    --------
    from statsmodels.api import datasets, tsa
    import pandas as pd
    dta = datasets.macrodata.load_pandas().data
    index = pd.DatetimeIndex(start='1959Q1', end='2009Q4', freq='Q')
    dta.set_index(index, inplace=True)
    
    #Run original tsa.filters two-sided hp filter
    cycle_tsa, trend_ts = tsa.filters.hpfilter(dta.realgdp, 1600)
    #Run two-sided implementation
    cycle2, trend2 = hprescott(dta.realgdp, 2, 1600)
    #Run one-sided implementation
    cycle1, trend1 = hprescott(dta.realgdp, 1, 1600)
    '''
    
    #Determine smooth if a specific frequency is given
    if freq == 'q':
        smooth = 1600 #quarterly
    elif freq == 'a':
        smooth = 6.25 #annually
    elif freq == 'm':
        smooth = 129600 #monthly
    elif freq != '':
        print('''Invalid frequency parameter inputted. Defaulting to defined smooth
        parameter value or 1600 if no value was provided.''')
    
    pw = PandasWrapper(X)
    X = array_like(X, 'X', ndim=1)
    T = len(X)
    
    #Preallocate trend array
    trend = np.zeros(len(X))

    #Rearrange the first order conditions of minimization problem to yield matrix
    #First and last two rows are mirrored
    #Middle rows follow same pattern shifting position by 1 each row

    a1 = np.array([1+smooth, -2*smooth, smooth])
    a2 = np.array([-2*smooth, 1+5*smooth, -4*smooth, smooth])
    a3 = np.array([smooth, -4*smooth, 1+6*smooth, -4*smooth, smooth])
    
    Abeg = np.concatenate(([np.append([a1],[0])],[a2]))
    Aend = np.concatenate(([a2[3::-1]], [np.append([0],[a1[2::-1]])]))
    
    Atot = np.zeros((T, T))
    Atot[:2,:4] = Abeg
    Atot[-2:,-4:] = Aend
	
    for i in range(2, T-2):
        Atot[i,i-2:i+3] = a3
	
    if (side == 1):
        t = 2
        trend[:t] = X[:t]

        # Third observation minimization problem is as follows	
        r3 = np.array([-2*smooth, 1+4*smooth, -2*smooth])
		
        Atmp = np.concatenate(([a1, r3], [a1[2::-1]]))
        Xtmp = X[:t+1]

        # Solve the system A*Z = X
        trend[t] = cho_solve(cho_factor(Atmp), Xtmp)[t]
		
        t += 1

        #Pattern begins with fourth observation
        #Create base A matrix with unique first and last two rows
        #Build recursively larger through time period
        Atmp = np.concatenate(([np.append([a1],[0])],[a2],[a2[3::-1]],[np.append([0],a1[2::-1])]))
        Xtmp = X[:t+1]

        trend[t] = cho_solve(cho_factor(Atmp), Xtmp)[t]
		
        while (t < T-1):
		
            t += 1
			
            Atmp = np.concatenate((Atot[:t-1,:t+1], np.zeros((2, t+1))))
            Atmp[t-1:t+1,t-3:t+1] = Aend

            Xtmp = X[:t+1]
            trend[t] = cho_solve(cho_factor(Atmp), Xtmp)[t]
		
    elif (side== 2):
        trend = cho_solve(cho_factor(Atot), X)
    else:
        raise ValueError('Side Parameter should be 1 or 2')

    cyclical = X - trend
    
    return pw.wrap(cyclical, append='cyclical'), pw.wrap(trend, append='trend')
