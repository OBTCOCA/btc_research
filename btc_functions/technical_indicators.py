import sys

if sys.platform == 'linux':
    sys.path.insert(1,'/home/oren/Research/tapiero')
else:
    sys.path.insert(1,'C:/Research/tapiero')

import pandas as pd 
import numpy as np 
import math

from tapiero.FUNCTIONS.WMA import WMA
from tapiero.FUNCTIONS.exponential_filters import exponential_filters

class techAnalysis(object):
    
    def __init__(self,Open,High,Low,Close):
        self.Open = Open
        self.High = High
        self.Low = Low
        self.Close = Close
        
    def hilbert_transform(self,x,L,Imult = 0.635,Qmult = 0.338):
        InPhase,Quadrature,Val = [np.zeros_like(x.values) for i in range(3)]

        for t in range(L,N):
            Val[t] = 10000*np.log(x.values[t]/x.values[t-L])
            InPhase[t] = 1.25*(Val[t-4] - Imult*Val[t-2]) + Imult*InPhase[t-3]
            Quadrature[t] = Val[t-2]-Qmult*Val[t]+Qmult*Quadrature[t-2]
        
        InPhase = pd.Series(InPhase,index = x.index)
        Quadrature = pd.Series(Quadrature,index = x.index)
        
        return InPhase,Quadrature

    def signal2noise(self,a1 = 0.2,a2 = 0.2,a3 = 0.25, L=7,Imult = 0.635,Qmult = 0.338):
        
        x = 0.5*(self.High + self.Low)
        
        InPhase,Quadrature = self.hilbert_transform(x,L,Imult,Qmult)

        Val2 = (InPhase**2 + Quadrature**2)
        Val2 = Val2.ewm(alpha = a1).mean()

        Range = 10000*np.log(self.High/self.Low)
        Range = Range.ewm(alpha = a2).mean()

        Val2.loc[Val2 < 0.001] = 0.001

        Amplitude = 10*np.log(Val2/Range**2)/np.log(10) + 1.9
        Amplitude = Amplitude.ewm(alpha = a3).mean()
        Amplitude.loc[Amplitude<0] = np.nan
        return Amplitude
        
    def inverse_fisher(self,x):
        return (np.exp(2*x)-1)/(np.exp(2*x)+1)
    
    def RSI(self,L,method = 'ema'):
        x = 0.5*(self.High + self.Low)
        
        delta = np.log(x).diff().fillna(0)
        dUp, dDown = delta.copy(), delta.copy()
        dUp[dUp < 0] = 0
        dDown[dDown > 0] = 0
    
        if method == 'ema':
            RolUp = dUp.ewm(span = L).mean()
            RolDown = dDown.ewm(span = L).mean().abs()
        else:
            RolUp = dUp.rolling(L).mean()
            RolDown = dDown.rolling(L).mean().abs()


        RS = RolUp / RolDown
        rsi = 100.0 - (100.0/(1.0+RS))
        return rsi
    
    def truncated_RSI(self,L,K):
        x = 0.5*(self.High + self.Low)
        
        delta = np.log(x).diff().fillna(0)
        dUp, dDown = delta.copy(), delta.copy()
        dUp[dUp < 0] = 0
        dDown[dDown > 0] = 0

        RolUp = dUp.rolling(K).apply(lambda x: x.ewm(span = L).mean()[-1])
        RolDown = dDown.rolling(K).apply(lambda x: x.ewm(span = L).mean()[-1]).abs()

        RS = RolUp / RolDown
        rsi = 100.0 - (100.0/(1.0+RS))
        return rsi   
    
    def normalize_rsi(self,x):
        return 0.1*(x-50)
    
    def smooth_inverse_fish_RSI(self,L,ZL):
        x = 0.5*(self.High + self.Low)
        
        SVE_RainbowAverage = ( 
            5 * WMA( x, 2 ) +
            4 * WMA( WMA( x, 2 ), 2 ) +
            3 * WMA( WMA( WMA( x, 2 ), 2 ), 2 ) + 
            2 * WMA( WMA( WMA( WMA( x, 2 ), 2 ) , 2 ), 2 ) + 
                WMA( WMA( WMA( WMA( WMA( x, 2 ),  2 ), 2 ), 2 ), 2 ) + 
                WMA( WMA( WMA( WMA( WMA(WMA( x, 2 ), 2 ), 2 ), 2 ), 2 ), 2 ) + 
                WMA( WMA( WMA( WMA( WMA( WMA( WMA( x, 2 ), 2 ), 2 ), 2 ), 2 ), 2 ), 2 ) +
                WMA( WMA( WMA( WMA( WMA( WMA( WMA( WMA( x, 2 ), 2 ), 2 ), 2 ), 2 ), 2 ), 2 ), 2 ) +
                WMA( WMA( WMA( WMA( WMA( WMA( WMA( WMA( WMA( x, 2 ), 2 ), 2 ), 2 ), 2 ), 2 ), 2 ), 2 ), 2 ) +
                WMA( WMA( WMA( WMA( WMA( WMA( WMA( WMA( WMA( WMA( x, 2 ), 2 ), 2 ), 2 ), 2 ), 2 ), 2 ), 2 ), 2 ), 2 )
            )/20
        
        rsi = self.RSI(L,method = 'ema')
        nrsi = self.normalize_rsi(rsi)
        
        ema1 = nrsi.ewm(span = ZL).mean()
        ema2 = ema1.ewm(span = ZL).mean()
        xzl = ema1+(ema1-ema2)
        smooth_rsi = 50*(self.inverse_fisher(xzl)+1)
        return smooth_rsi


    def sinwave(self,L = 36,lowerband = 9,output = 1):
        S = 0.5*(self.High + self.Low)

            
        N = len(S)
        x = S.values

        angle = 2*math.pi/L
        alpha1 =  (1-math.sin(angle))/math.cos(angle)

        ang = math.sqrt(2)*math.pi/lowerband
        a1 = math.exp(-ang)
        b1 = 2*a1*math.cos(ang)
        c2 = b1
        c3 = -a1*a1
        c1 = 1-c2-c3

        HP,filt,wave,pwr,sinwave = [np.zeros_like(x) for i in range(5)]

        for i in range(2,N):
            HP[i] = 0.5*(1-alpha1)*(x[i] - x[i-1])+alpha1*HP[i-1]
            filt[i] = 0.5*c1*(HP[i] + HP[i-1]) + c2*filt[i-1] + c3*filt[i-2]
            wave[i] = (filt[i]+filt[i-1]+filt[i-2])/3
            pwr[i] = (filt[i]*filt[i]+filt[i-1]*filt[i-1]+filt[i-2]*filt[i-2])/3
            sinwave[i] = wave[i]/math.sqrt(pwr[i])

        out = pd.DataFrame(data = np.column_stack([wave,pwr,sinwave]),
                           index = S.index,
                           columns = ['wave','pwr','sinwave'])
        if output == 1:
            return out
        else:
            return out['sinwave']

    def autocorrelation(self,L,lev = 0):
        S = 0.5*(self.High + self.Low)


        Rt = 10000*S.groupby(S.index.strftime('%Y-%U')).apply(lambda x: np.log(S).diff().fillna(0))
        Rt_lag = 10000*S.groupby(S.index.strftime('%Y-%U')).apply(lambda x: np.log(x).diff().shift(1).fillna(0))
        
        if lev == 0:
            ac = Rt.ewm(span = L).corr(Rt_lag)
        elif lev == 1:
            ac = Rt.ewm(span = L).corr(Rt_lag.abs())
        elif lev == 2:
            ac = Rt.abs().ewm(span = L).corr(Rt_lag)
            
        return ac
            
    def ATR(self,L,method = 'sma'):
        
        HL = 10000*np.log(self.High/self.Low)
        HC = 10000*np.log(self.High/self.Close).abs()
        LC = 10000*np.log(self.Low/self.Close).abs()
        
        TR = pd.concat([HL,HC,LC],axis = 1).max(axis = 1)

        if method == 'sma':
            ATR = TR.rolling(L).mean()
        else:
            ATR = TR.ewm(span = L).mean()

        return ATR       

    def stochastic_K(self,L):
        K = (self.Close-self.Low.rolling(L).min())/(self.High.rolling(L).max()-self.Low.rolling(L).min())
        return 100*K
    
    def williamsR(self,L):
        R = (self.High.rolling(L).max()-self.Close)/(self.High.rolling(L).max()-self.Low.rolling(L).min())
        return -100*R
        
    def MACD(self,Ls,Ll):
        Px = 0.5*(self.High+self.Low)
        
        if Ls>Ll:
            MACD = Px.ewm(span = Ls).mean()-Px.ewm(span = Ll).mean()
        else:
            MACD = Px.ewm(span = Ll).mean()-Px.ewm(span = Ls).mean()
        return MACD
        
    def ROC(self,L):
        Px = 0.5*(self.High+self.Low)
        return 10000*np.log(Px/Px.shift(L))
        
