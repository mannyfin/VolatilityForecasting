import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as lr
from Performance_Measure import *

def lin_reg(data, n, warmup_period):
# def lin_reg(data, n, filename, stringinput, warmup_period):
    """
    :param warmup_period uses a fixed window warmup period defined by the var, warmup_period
    :param data could be train_sample or test_sample
    :param n is the number of regressors
    :return: MSE, QL, ln(SE) and parameters b and c
    """
    # use log volatility rather than volatility for linear regression model
    LogVol = np.log(data['Volatility_Time'].astype('float64'))
    PredictedLogVol=[]
    x = [i for i in range(n)]
    for initial in range(warmup_period, len(LogVol)):
        for i in range(n):
            x[i] = LogVol[i:(initial +i-n-1)]

        xstacked = np.column_stack(x)

        y = LogVol[n:initial-1]
        A = lr()
        A.fit(xstacked, y)
        b = [A.coef_[i] for i in range(n)]
        c = A.intercept_

        # reshape data for prediction
        PredictedLogVol.append(A.predict(LogVol[initial-n : initial].values.reshape(1, -1))[0])


    y = data.Volatility_Time[warmup_period:]

    PredictedVol = pd.Series(np.exp(PredictedLogVol))
    Performance_ = PerformanceMeasure()
    MSE = Performance_.mean_se(observed=y, prediction=PredictedVol)
    QL = Performance_.quasi_likelihood(observed=y.astype('float64'),
                                       prediction=PredictedVol.astype('float64'))

    # dates = data['Date'][n:]
    # label=str(filename)+" "+str(stringinput)+" Linear Regression: "+str(n) + " Past Vol SE "
    # SE(y, PredictedVol, dates,function_method=label)

    SE = [(y.values[i] - PredictedVol.values[i]) ** 2 for i in range(len(y))]
    ln_SE = pd.Series(np.log(SE))

    return MSE, QL, ln_SE, b, c