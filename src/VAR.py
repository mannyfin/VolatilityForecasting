# This file implements VAR (vector auto regression) desceibed in the paper
# Wilms, Ines, Jeroen Rombouts, and Christophe Croux. "Lasso-based forecast combinations for forecasting realized variances." (2016).

import numpy as np
from sklearn.linear_model import LinearRegression as lr
import pandas as pd
from Performance_Measure import *

Vol_df = [daily_vol_combined,weekly_vol_combined,monthly_vol_combined]

LogRV_df = [np.log(Vol_df[i]**2) for i in range(3)]

daily_warmup_lag_1 = 400 # warm-up period is 400 for daily
daily_warmup_lag_2 = 400 # warm-up period is 400 for daily
daily_warmup_lag_3 = 400 # warm-up period is 400 for daily

p=[1,2,3] # p is lag, picking 1,2 and 3 according to Amin's suggesting

# p = 6 # p is lag, according to BIC value produced in R
# q= 9 # q is the number of currency pairs
# weekly_warmup = 70 # warm-up period is 400 for weekly
# monthly_warmup = 25 # warm-up period is 400 for monthly

# t = daily_warmup
# n1 = len(Daily_Vol_df)-daily_warmup # n1 is the sample size for daily VAR regression
# n2 = len(Daily_Vol_df)-daily_warmup # n2 is the sample size for weekly VAR regression
# n3 = len(Daily_Vol_df)-daily_warmup # n3 is the sample size for monthly VAR regression


LogRV_df = np.log(Daily_Vol_df**2)

p = 6 # p is lag, according to BIC value produced in R
q= 9 # q is the number of currency pairs
daily_warmup = 400 # warm-up period is 400 for daily
# t = daily_warmup
n = len(Daily_Vol_df)-daily_warmup # n is the sample size for linear regression

'''
    Construct y as a list of 9 lists.
    The k-th list inside y is a series of logRV for the k-th currency pair for k=1,2,...,9
'''
def get_y(q, p, t,n):
    '''
    :param q: q=9 in this project since we have 9 currency pairs
    :param p: p is lag
    :param t: t = warm-up period
    :param n: n = len(Vol_df)-warmup
    :return: y as inputs into LR for all currency pairs
    '''
    y = []
    for i in range(q):
        y_i= []
        for k in range(n): # n is the sample size
            y_i.append( LogRV_df.iloc[t+k][i] )
        y.append(y_i)
    return y

def x_mat_t_n_qp(q, p, t,n):
    '''
    :param q: q=9 in this project since we have 9 currency pairs
    :param p: p is lag
    :param t: t = warm-up period
    :param n: n = len(Vol_df)-warmup
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
def predictlogRV(q,p,t,n):
    '''
    :param q: q=9 in this project since we have 9 currency pairs
    :param p: p is lag
    :param t: t = warm-up period
    :param n: n = len(Vol_df)-warmup
    :return: the predicted logRV for all 9 currency pairs
    '''
    n = len(Daily_Vol_df)-daily_warmup
    x = x_mat_t_n_qp(q=9, p=p, t=t,n=n)
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
def VAR_MSE_QL(q,p,t,n):
    '''

    :param y: realized logRV for all currency pairs
    :param q: q=9 in this project since we have 9 currency pairs
    :param p: p is lag
    :param t: t = warm-up period
    :param n: n = len(Vol_df)-warmup
    :return: MSE, QL and SE plot
    '''
    n = len(Daily_Vol_df)-daily_warmup
    y = get_y(q, p, t,n)
    PredictedlogRVforAll = predictlogRV(q=q, p=p, t=t,n=n)
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
    return MSEforAll, QLforAll

def VAR_MSE_QL(q, p, t, n, data): # can combine with MSE and QL
    n = len(Daily_Vol_df) - daily_warmup
    PredictedlogRVforAll = predictlogRV(q=q, p=p, t=t,n=n)
    label = str(filename) + " " + str(stringinput) + " VAR"
    # TODO: change this label and SE below
    for i in range(q):
        SE(np.sqrt(np.e(y[i])), np.sqrt(np.e(PredictedlogRVforAll[i])), dates, function_method=label)








#
#
# '''
#     Construct y as a list of 9 lists.
#     The k-th list inside y is a series of logRV for the k-th currency pair for k=1,2,...,9
# '''
# def get_y(q, p, t,n,indicator):
#     '''
#     :param q: q=9 in this project since we have 9 currency pairs
#     :param p: p is lag
#     :param t: t = warm-up period
#     :param n: n = len(Vol_df)-warmup
#     :param indicator: indicator is a string input, which can be "Daily", "Weekly" and "Monthly"
#     :return: y as inputs into LR for all currency pairs
#     '''
#     if indicator=="Daily":
#         indic = 0
#     elif indicator=="Weekly":
#         indic = 1
#     elif indicator=="Monthly":
#         indic = 2
#
#     y = []
#     for i in range(q):
#         y_i= []
#         for k in range(n): # n is the sample size
#             y_i.append( LogRV_df[indic].iloc[t+k][i] )
#         y.append(y_i)
#     return y
#
# def x_mat_t_n_qp(q, p, t,n, indicator):
#     '''
#     :param q: q=9 in this project since we have 9 currency pairs
#     :param p: p is lag
#     :param t: t = warm-up period
#     :param n: n = len(Vol_df)-warmup
#     :param indicator: indicator is a string input, which can be "Daily", "Weekly" and "Monthly"
#     :return: the x matrix as a input for regression, where the dimension of x is n*(qp)
#     '''
#     if indicator=="Daily":
#         indic = 0
#     elif indicator=="Weekly":
#         indic = 1
#     elif indicator=="Monthly":
#         indic = 2
#
#     x =  pd.DataFrame()
#     for m in range(n):
#         x_t_vec = []
#         for k in range(q):
#             for i in range(1,p+1):
#                 x_t_vec.append(LogRV_df[indic].iloc[t+m-i][k])
#         x = x.append([x_t_vec])
#     return x
# '''
#      Fitting parameters and making prediction based on fitted models
#      PredictedlogRV collects the predicted logRV for all 9 currency pairs
#
# '''
# def predictlogRV(q,p,t,n,indicator):
#     '''
#     :param q: q=9 in this project since we have 9 currency pairs
#     :param p: p is lag
#     :param t: t = warm-up period
#     :param n: n = len(Vol_df)-warmup
#     :param indicator: indicator is a string input, which can be "Daily", "Weekly" and "Monthly"
#     :return: the predicted logRV for all 9 currency pairs
#     '''
#     # n = len(Daily_Vol_df)-daily_warmup
#     x = x_mat_t_n_qp(q=9, p=p, t=t,n=n, indicator=indicator)
#     PredictedlogRVforAll = []
#     for i in range(9):
#         A = lr()
#         A.fit( x, y[i] )
#         b = A.coef_
#         c = A.intercept_
#         PredictedlogRV = []
#         for k in range(n):
#             PredictedlogRV.append( A.predict( x.iloc[k].values.reshape(1, -1) )[0] )
#         PredictedlogRVforAll.append(PredictedlogRV)
#     return PredictedlogRVforAll
#
#
# '''
#     Obtaining MSE and QL
# '''
# def VAR_MSE_QL(q,p,t,n,indicator):
#     '''
#
#     :param y: realized logRV for all currency pairs
#     :param q: q=9 in this project since we have 9 currency pairs
#     :param p: p is lag
#     :param t: t = warm-up period
#     :param n: n = len(Vol_df)-warmup
#     :param indicator: indicator is a string input, which can be "Daily", "Weekly" and "Monthly"
#     :return: MSE, QL and SE plot
#     '''
#     # n = len(Daily_Vol_df)-daily_warmup
#     y = get_y(q, p, t,n, indicator)
#     PredictedlogRVforAll = predictlogRV(q=q, p=p, t=t,n=n, indicator=indicator)
#     Performance_ = PerformanceMeasure()
#     MSEforAll = []
#     QLforAll = []
#
#     for i in range(q):
#         MSE = Performance_.mean_se(observed=np.sqrt(np.exp(y[i])), prediction=np.sqrt(np.exp(PredictedlogRVforAll[i])))
#         QL = Performance_.quasi_likelihood(observed=np.sqrt(np.exp(y[i])), prediction=np.sqrt(np.exp(PredictedlogRVforAll[i])))
#         MSEforAll.append(MSE)
#         QLforAll.append(QL)
#     return MSEforAll, QLforAll
#
# '''
#     Obtaining SE plot
# '''
#
# def VAR_SE(q, p, t, n, indicator, data): # can combine with MSE and QL
#     # n = len(Daily_Vol_df) - daily_warmup
#     # dates = data['Date'][n:]
#     # TODO: take care of dates
#     PredictedlogRVforAll = predictlogRV(q=q, p=p, t=t,n=n)
#     label = str(filename) + " " + str(stringinput) + " VAR"
#     # TODO: change this label and SE below
#     for i in range(q):
# #         SE(np.sqrt(np.e(y[i])), np.sqrt(np.e(PredictedlogRVforAll[i])), dates, function_method=label)Daily_Vol_df = daily_vol_combined
#
#
# MSE_daily,QL_daliy = VAR_MSE_QL(q=9, p=6, t=daily_warmup, n=len(LogRV_df[0])-daily_warmup,indicator="Daily")
# MSE_weekly,QL_weekly = VAR_MSE_QL(q=9, p=6, t=weekly_warmup, n=len(LogRV_df[1])-weekly_warmup,indicator="Weekly")
# MSE_monthly,QL_monthly = VAR_MSE_QL(q=9, p=6, t=monthly_warmup, n=len(LogRV_df[2])-monthly_warmup,indicator="Monthly")
