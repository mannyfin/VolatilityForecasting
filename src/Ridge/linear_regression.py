import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as lr
from Performance_Measure import *


def lin_reg(data, n, warmup_period, test=False):
    # TODO: write functions to find the optimal number of regressors n in the training set and collect MSE, QL and ln(SE) in the test set
    # def lin_reg(data, n, filename, stringinput, warmup_period):
    """
    :param warmup_period uses a fixed window warmup period defined by the var, warmup_period
    :param data could be train_sample or test_sample
    :param n is the number of regressors
    :return: MSE, QL, ln(SE) and parameters b and c
    """
    param_list=[]


    # use log volatility rather than volatility for linear regression model
    LogVol = np.log(data['Volatility_Time'].astype('float64'))
    PredictedLogVol=[]
    x = [i for i in range(n)]
    for initial in range(warmup_period, len(LogVol)):
        for i in range(n):
            x[i] = LogVol[i:(initial + i-n-1)]

        xstacked = np.column_stack(x)

        y = LogVol[n:initial-1]
        A = lr()
        A.fit(xstacked, y)
        b = [A.coef_[i] for i in range(n)]
        c = A.intercept_

        # reshape data for prediction
        PredictedLogVol.append(A.predict(LogVol[initial-n : initial].values.reshape(1, -1))[0])
        SE = (A.predict(LogVol[initial-n : initial].values.reshape(1, -1)) - LogVol[initial]) ** 2
        param_list.append([b + [c] + [SE] ][0])

    # plot the regressors and intercept
    param_plot = pd.DataFrame(np.array(param_list), index=data.Date[warmup_period:],
                              columns=['b' + str(count) for count, elem in enumerate(b)] + ['c'] + ['SE'])

    param_plot.plot(title='Regressors and Intercept for n=' + str(n), figsize=(9, 6))\
              .legend(loc="center left", bbox_to_anchor=(1, 0.5))


    if test is False:
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

    elif test[0] is True:
        # test[1] is the test set. First convert to log vol
        y = np.log(test[1].Volatility_Time.astype('float64'))
        # take the last n predicted elements (starting from the beginning of the test set till the end)
        # and put them in PredictedVol
        tested_vol=[]
        # use the last n samples of the train set for predicting the first value of the test set
        test_set = pd.concat([LogVol[-n:],y],axis=0)
        for initial in range(0, len(y)):
                tested_vol.append(A.predict(test_set.iloc[initial: initial+n].values.reshape(1, -1))[0])
        # PredictedVol = pd.Series(np.exp(PredictedLogVol[-len(test[1]):]))

        tested_vol = pd.Series(np.exp(tested_vol),index=y.index)
        y = y.apply(np.exp)
        # y = np.exp(y.reset_index().drop('index',axis=1))
        # y.drop('index',axis=1)
        Performance_ = PerformanceMeasure()
        MSE = Performance_.mean_se(observed=y, prediction=tested_vol)
        QL = Performance_.quasi_likelihood(observed=y.astype('float64'),
                                           prediction=tested_vol.astype('float64'))

        # dates = data['Date'][n:]
        # label=str(filename)+" "+str(stringinput)+" Linear Regression: "+str(n) + " Past Vol SE "
        # SE(y, PredictedVol, dates,function_method=label)

        SE = (y - tested_vol) ** 2
        # SE = [(y.values[i] - tested_vol.values[i]) ** 2 for i in range(len(y))]
        ln_SE = pd.Series(np.log(SE))

        return MSE, QL, ln_SE,tested_vol, b, c



