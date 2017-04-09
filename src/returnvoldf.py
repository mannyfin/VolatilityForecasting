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
    combined = pd.concat([vol.Date[:-1], ret.Return_Time[:-1], vol.Volatility_Time.iloc[:-1]
                         , vol.Volatility_Time.shift(-1)], axis=1).dropna().set_index('Date')
    preprocess = v.join(combined, how='inner')
    preprocess.columns = ['V', 'ret_now',  'vol_now', 'vol_future']
    # make label column
    preprocess['volxret_now'] = preprocess.vol_now*preprocess.ret_now
    preprocess['label'] = (preprocess.vol_future > preprocess.vol_now).astype(int)
    # does not give setting with copy warning
    preprocess.loc[preprocess.label == 0, 'label'] = -1

    test_sample = preprocess[preprocess.value == 0]
    train_sample = preprocess[preprocess.value == 1]

    return preprocess,test_sample, train_sample