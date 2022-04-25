import sys
import pandas as pd

def get_data(pair,per = '1M'):
    if sys.platform == 'linux':
        path = '/home/oren/Data/OHLCM7/'
    else:
        path = '/Users/orentapiero/Data/'
        
    if per == '1M':
        ohlc=pd.read_csv(path+pair+'_1T_'+'ohlc.csv')
        ohlc.index = pd.to_datetime(ohlc['TIMESTAMP'])
    else:
        ohlc=pd.read_csv(path+pair+'_5T_'+'ohlc.csv')
        ohlc.index = pd.to_datetime(ohlc['TIMESTAMP'])
    
    del ohlc['TIMESTAMP']
    
    Mid = ohlc[['mid_open','mid_high','mid_low','mid_close','msg']]
    Ask = ohlc[['ask_open','ask_high','ask_low','ask_close']]
    Bid = ohlc[['bid_open','bid_high','bid_low','bid_close']]

    Ask = Ask.groupby(Ask.index).last()
    Bid = Bid.groupby(Bid.index).last()
    Mid = Mid.groupby(Mid.index).last()

    Mid = Mid.rename(columns = {'mid_open':'open','mid_high':'high','mid_low':'low','mid_close':'close'})
    Mid = Mid.loc[Mid.index.day_name() != 'Sunday']

    Ask = Ask.rename(columns = {'ask_open':'open','ask_high':'high','ask_low':'low','ask_close':'close'})
    Bid = Bid.rename(columns = {'bid_open':'open','bid_high':'high','bid_low':'low','bid_close':'close'})

    x = pd.concat([Ask.close.rename('Ask'),
                   Bid.close.rename('Bid'),
                   Mid.open.rename('open'),
                   Mid.high.rename('high'),
                   Mid.low.rename('low'),
                   Mid.close.rename('close'),
                   Mid.msg.rename('msg')],axis = 1)
    return x
