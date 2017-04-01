import pandas as pd

def retvoldf(ret, vol, v):
    """
    
    :param ret: returns (pd.dataframe)
    :param vol: vol (pd.dataframe)
    :param v: v is from v.csv given to us
    :return: Dataframe 
    """

    # this line below creates a DataFrame obj that intersects the dates from vol and v.csv. It then arranges the vols in
    # a time based fashion
    combined = pd.concat([vol.Date[1:-1], ret.Return_Time[:-2], vol.Volatility_Time[:-2], vol.Volatility_Time[:-1]
                         .shift(-1), vol.Volatility_Time[:].shift(-2)], axis=1).dropna().set_index('Date')
    preprocess = v.join(combined, how='inner')
    preprocess.columns = ['V', 'ret_past', 'vol_past', 'vol_now', 'vol_future']

    return preprocess