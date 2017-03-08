# This file implements VAR (vector auto regression) desceibed in the paper
# Wilms, Ines, Jeroen Rombouts, and Christophe Croux. "Lasso-based forecast combinations for forecasting realized variances." (2016).

import numpy as np
from sklearn.linear_model import LinearRegression as lr
import pandas as pd
from Performance_Measure import *

Daily_Vol_df = pd.DataFrame(np.random.randn(1000, 9)+10) # experiment df
LogRV_df = np.log(Daily_Vol_df**2)

p = 6 # p is lag, according to BIC value produced in R
q= 9 # q is the number of currency pairs
daily_warmup = 40 # warm-up period is 400 for daily
# t = daily_warmup
n = len(Daily_Vol_df)-daily_warmup # n is the sample size for linear regression

'''
    Construct y as a list of 9 lists.
    The k-th list inside y is a series of logRV for the k-th currency pair for k=1,2,...,9
'''
def get_y(q, p, t):
    '''
    :param q: q=9 in this project since we have 9 currency pairs
    :param p: p is lag
    :param t: t = warm-up period
    :return: y as inputs into LR for all currency pairs
    '''
    y = []
    for i in range(q):
        y_i= []
        for k in range(n): # n is the sample size
            y_i.append( LogRV_df.iloc[t+k][i] )
        y.append(y_i)
    return y

def x_mat_t_n_qp(q, p, t):
    '''
    :param q: q=9 in this project since we have 9 currency pairs
    :param p: p is lag
    :param t: t = warm-up period
    :return: the x matrix as a input for regression, where the dimension of x is n*(qp)
    '''
    x =  pd.DataFrame()
    for m in range(n):
        x_t_vec = []
        for k in range(q):
            for i in range(1,p+1):
                x_t_vec.append(LogRV_df.iloc[t+m-i][k])
        x = x.append([x_t_vec])
    return x
'''
     Fitting parameters and making prediction based on fitted models
     PredictedlogRV collects the predicted logRV for all 9 currency pairs

'''
def predictlogRV(q,p,t):
    '''
    :param q: q=9 in this project since we have 9 currency pairs
    :param p: p is lag
    :param t: t = warm-up period
    :return: the predicted logRV for all 9 currency pairs
    '''
    x = x_mat_t_n_qp(q=9, p=p, t=t)
    PredictedlogRVforAll = []
    for i in range(9):
        A = lr()
        A.fit( x, y[i] )
        b = A.coef_
        c = A.intercept_
        PredictedlogRV = []
        for k in range(n):
            PredictedlogRV.append( A.predict( x.iloc[k].values.reshape(1, -1) )[0] )
        PredictedlogRVforAll.append(PredictedlogRV)
    return PredictedlogRVforAll


'''
    Obtaining MSE and QL
'''
def VAR_Performance(q,p,t,data):
    '''

    :param y: realized logRV for all currency pairs
    :param q: q=9 in this project since we have 9 currency pairs
    :param p: p is lag
    :param t: t = warm-up period
    :return: MSE, QL and SE plot
    '''
    y = get_y(q, p, t)
    PredictedlogRVforAll = predictlogRV(q=q, p=p, t=t)
    Performance_ = PerformanceMeasure()
    MSEforAll = []
    QLforAll = []
    # dates = data['Date'][n:]
    # TODO: take care of dates
    for i in range(q):
        MSE = Performance_.mean_se(observed=np.sqrt(np.exp(y[i])), prediction=np.sqrt(np.exp(PredictedlogRVforAll[i])))
        QL = Performance_.quasi_likelihood(observed=np.sqrt(np.exp(y[i])), prediction=np.sqrt(np.exp(PredictedlogRVforAll[i])))
        MSEforAll.append(MSE)
        QLforAll.append(QL)

        # label = str(filename) + " " + str(stringinput) + " VAR"
        # TODO: change this label and SE below
        # SE(np.sqrt(np.e(y[i])), np.sqrt(np.e(PredictedlogRVforAll[i])), dates, function_method=label)

    return MSEforAll, QLforAll