import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge as bayesian_ridge
from Performance_Measure import *


def bayes_ridge_reg(data, n, warmup_period,alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, test=False):
    # TODO: write functions to find the optimal number of regressors n in the training set and collect MSE, QL and ln(SE) in the test set
    """
    :param warmup_period uses a fixed window warmup period defined by the var, warmup_period
    :param data could be train_sample or test_sample
    :param n is the number of regressors
    :return: MSE, QL, ln(SE) and parameters b and c
    lamda = L2 penalty term, in sklearn docs this is alpha
    test: False if doing training. If you are doing testing, pass a tuple with (True, test_set) where test_set is pref
         a dataframe.
    """
    # use log volatility rather than volatility for linear regression model
    param_list=[]
    # preprocess
    vol = data.copy()
    # separate date out and remove date from train_set
    Date = data.Date
    vol.drop('Date', axis=1, inplace=True)
    # take log vol
    LogVol = vol.apply(np.log)
    # split up data
    LogVol = LogVol.T
    LogVol = LogVol.unstack()

    predict_set = vol.apply(np.log)

    # use log volatility rather than volatility for linear regression model
    # LogVol = np.log(data['Volatility_Time'].astype('float64'))

    PredictedLogVol=[]

    #     too many values to test so not worth plotting and saving all the figs of the hyperparameters
    x = [i for i in range(n)]
    # for initial in range(warmup_period, len(LogVol)):
    for initial in range(warmup_period, LogVol.last_valid_index()[0]+1):
        for i in range(n):
            # x[i] = LogVol[i:(initial + i-n-1)]
            x[i] = LogVol.loc[i:(initial + i-n-1)]

        xstacked = np.column_stack(x)

        y = LogVol.loc[n:initial-1]
        A = bayesian_ridge(alpha_1=alpha_1, alpha_2=alpha_2, compute_score=False,
                           copy_X=True, fit_intercept=True, lambda_1=lambda_1, lambda_2=lambda_2,
                           normalize=False, tol=0.000001, verbose=False)

        A.fit(xstacked,  y.loc[:,:])
        b = [A.coef_[i] for i in range(n)]
        c = A.intercept_

        # reshape data for prediction
        # PredictedLogVol.append(A.predict(LogVol.loc[initial-n : initial].values.reshape(1, -1))[0])
        PredictedLogVol.append(A.predict(predict_set.loc[initial-n: initial-1].T))
        SE = (A.predict(predict_set.loc[initial-n : initial-1].T) - predict_set.loc[initial].T) ** 2
        # SE = (A.predict(LogVol.loc[initial-n+1 : initial].values.reshape(1, -1)) - LogVol.loc[initial]) ** 2
        param_list.append([b + [c] + [SE] ][0])

    if test is False:
        y = vol[warmup_period:]

        PredictedVol = pd.DataFrame(np.exp(PredictedLogVol))
        Performance_ = PerformanceMeasure()
        MSE = Performance_.mean_se(observed=y, prediction=PredictedVol)
        QL = Performance_.quasi_likelihood(observed=y.astype('float64'),
                                           prediction=PredictedVol.astype('float64'))

        # dates = data['Date'][n:]
        # label=str(filename)+" "+str(stringinput)+" Linear Regression: "+str(n) + " Past Vol SE "
        # SE(y, PredictedVol, dates,function_method=label)

        SE = [(y.values[i] - PredictedVol.values[i]) ** 2 for i in range(len(y))]
        ln_SE = pd.DataFrame(np.log(SE))

        return MSE, QL, ln_SE, b, c

    # this should be the proper code regardless because all we need for the test set is A.predict

    elif test[0] is True:

        # test[1] is the test set. First convert to log vol
        test_Date = test[1].Date

        test_vols = test[1].copy()
        test_vols.drop('Date', axis=1, inplace=True)
        # take log vol
        y = test_vols.apply(np.log).astype('float64')


        # y = np.log(test[1].Volatility_Time.astype('float64'))
        # take the last n predicted elements (starting from the beginning of the test set till the end)
        # and put them in PredictedVol
        tested_vol=[]
        # use the last n samples of the train set for predicting the first value of the test set
        # test_set = pd.concat([LogVol[-n:],y],axis=0)
        test_set = pd.concat([predict_set[-n:],y],axis=0).reset_index(drop=True)

        for initial in range(0, len(y)):
                tested_vol.append(A.predict(test_set.loc[initial: initial+n-1].T))
        # PredictedVol = pd.Series(np.exp(PredictedLogVol[-len(test[1]):]))

        tested_vol = pd.DataFrame(np.exp(tested_vol),index=y.index)
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

        # was having issues with subtraction, so just made the column headers the same
        tested_vol.columns = y.columns

        SE = (y - tested_vol) ** 2
        # SE = [(y.values[i] - tested_vol.values[i]) ** 2 for i in range(len(y))]
        ln_SE = pd.DataFrame(np.log(SE))

        return MSE, QL, ln_SE, tested_vol
