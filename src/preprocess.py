import pandas as pd

def preprocess_data(ret, vol, v):
    """
    
    :param ret: returns (pd.dataframe)
    :param vol: vol (pd.dataframe)
    :param v: v is from v.csv given to us
    :return: Dataframe 
    """

    # this line below creates a DataFrame obj that intersects the dates from vol and v.csv. It then arranges the vols in
    # a time based fashion
    combined = pd.concat([vol.Date, ret.Return_Time, vol.Volatility_Time], axis=1).dropna().set_index('Date')
    preprocess = v.join(combined, how='inner')
    # preprocess.columns = ['V', 'ret_time', 'vol_time']
    # make label column
    preprocess['label'] = (preprocess.Volatility_Time > preprocess.Volatility_Time.shift(1)).astype(int)
    # does not give setting with copy warning
    preprocess.loc[preprocess.label == 0, 'label'] = -1
    test_sample = preprocess[preprocess.value == 0]
    train_sample = preprocess[preprocess.value == 1]

    return preprocess[1:], test_sample, train_sample