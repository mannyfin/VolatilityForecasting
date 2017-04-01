import pandas as pd
import numpy as np
from SEplot import se_plot as SE

def test(ret, vol):
    # given v.csv for the hw
    v = pd.read_csv('v.csv')
    v.columns = ['Date', 'value']
    v = v.set_index('Date')

    combined = pd.concat([vol.Date[1:-1], ret.Return_Time[:-2], vol.Volatility_Time[:-2], vol.Volatility_Time[:-1]
                         .shift(-1), vol.Volatility_Time[:].shift(-2)], axis=1).dropna().set_index('Date')
    preprocess = v.join(combined, how='inner')
    print("hi")
