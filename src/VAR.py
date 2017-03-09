# This file implements VAR (vector auto regression) desceibed in the paper
# Wilms, Ines, Jeroen Rombouts, and Christophe Croux. "Lasso-based forecast combinations for forecasting realized variances." (2016).

# We only implements VAR on daily volatility data.
# We take log of annualized volatility to get logRV.
# We divide the data into training sample (2/3) and test sample (1/3).
# For lag p = 1, warm-up period/look-back window is 100.
# For lag p = 2, warm-up period/look-back window is 200.
# For lag p = 3, warm-up period/look-back window is 300.
# From the training sample, we get the best p giving the smallest MSE;
# Using the optimal p, we fit the model and make predictions on the test sample.
# We used a rolling window method rather than a growing window method.


import numpy as np
from sklearn.linear_model import LinearRegression as lr
import pandas as pd
from Performance_Measure import *

'''
    Construct y as a list of 9 lists.
    The k-th list inside y is a series of logRV for the k-th currency pair for k=1,2,...,9
'''
def get_y(LogRV_df,q, p, t,n):
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

def x_mat_t_n_qp(LogRV_df,q, p, t,n):
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
def predictlogRV(LogRV_df,q,p,t,n):
    '''
    :param q: q=9 in this project since we have 9 currency pairs
    :param p: p is lag
    :param t: t = warm-up period
    :param n: n = len(Vol_df)-warmup
    :return: the predicted logRV for all 9 currency pairs
    '''
    x = x_mat_t_n_qp(LogRV_df,q, p, t,n)
    y = get_y(LogRV_df,q, p, t,n)
    PredictedlogRVforAll = []
    for i in range(q):
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
def VAR_MSE_QL(LogRV_df,q,p,t,n):
    '''

    :param y: realized logRV for all currency pairs
    :param q: q=9 in this project since we have 9 currency pairs
    :param p: p is lag
    :param t: t = warm-up period
    :param n: n = len(Vol_df)-warmup
    :return: MSE, QL and SE plot
    '''
    y = get_y(LogRV_df,q, p, t,n)
    PredictedlogRVforAll = predictlogRV(LogRV_df,q, p, t,n)
    Performance_ = PerformanceMeasure()
    MSEforAll = []
    QLforAll = []
    for i in range(q):
        MSE = Performance_.mean_se(observed=np.exp(y[i]), prediction=np.exp(PredictedlogRVforAll[i]))
        QL = Performance_.quasi_likelihood(observed=np.exp(y[i]), prediction=np.exp(PredictedlogRVforAll[i]))
        MSEforAll.append(MSE)
        QLforAll.append(QL)
        mean_MSE = np.mean(MSEforAll)
        mean_QL = np.mean(QLforAll)

    return mean_MSE,mean_QL, MSEforAll, QLforAll

'''
    Obtaining squared errors
'''

def VAR_SE(q, p, t, n, data): # can combine with MSE and QL
    # TODO: input correct data input
    PredictedlogRVforAll = predictlogRV(LogRV_df,q, p, t,n)
    y = get_y(LogRV_df,q, p, t,n)
    label = str(filename) + " " + str(stringinput) + " VAR"
    # TODO: change this label
    for i in range(q):
        SE(np.sqrt(np.e(y[i])), np.sqrt(np.e(PredictedlogRVforAll[i])), dates, function_method=label)
        # TODO: get dates
        # TODO: add return

'''
    Obtaining optimal p
'''

Daily_Vol_df = daily_vol_combined
LogRV_df = np.log(Daily_Vol_df)
p = [1, 2, 3]  # p is lag, picking 1,2 and 3 according to Amin's suggesting
q = 9  # q is the number of currency pairs
daily_warmup = [100*p[i] for i in range(len(p))] # warm-up period is 100,200 and 300 respectively for daily data when p=1,2 and 3
len_training = int(2 / 3 * len(Daily_Vol_df))
n = [] # n collects the sample size for daily VAR regression when p=1, p=2 and p=3
for i in range(len(p)):
    n.append(len_training - daily_warmup[i])
VAR_p1= VAR_MSE_QL(LogRV_df,q=q,p=p[0],t=daily_warmup[0],n=n[0])
VAR_p2= VAR_MSE_QL(LogRV_df,q=q,p=p[1],t=daily_warmup[1],n=n[1])
VAR_p3= VAR_MSE_QL(LogRV_df,q=q,p=p[2],t=daily_warmup[2],n=n[2])

MSEs = [VAR_p1[0],VAR_p2[0],VAR_p3[0]]
QLs = [VAR_p1[1],VAR_p2[1],VAR_p3[1]]

optimal_p_MLE = MSEs.index(min(MSEs))+1 # the optimal p according to MLE criterion is 3
optimal_p_QL = QLs.index(min(QLs))+1 # the optimal p according to QL criterion is 3 as well
optimal_p = optimal_p_MLE # we use the optimal p according to MLE criterion as the optimal p

'''
    Using the optimal p on the test sample
'''
MSE_QL_optimal_p = VAR_MSE_QL(LogRV_df,q=q,p=p[2],t=len_training,n=len(Daily_Vol_df)-len_training)
MSE_optimal_p_avg = MSE_QL_optimal_p[0] # average MSE of 9 currency pairs
QL_optimal_p_avg = MSE_QL_optimal_p[1]  # average QL of 9 currency pairs
MSE_optimal_p_forAll = MSE_QL_optimal_p[2] # MSE of all 9 currency pairs
QL_optimal_p_forAll = MSE_QL_optimal_p[3]  # QL of all 9 currency pairs


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
#     PredictedlogRVforAll = predictlogRV(q=q, p=p, t=t,n=n)
#     label = str(filename) + " " + str(stringinput) + " VAR"
#     for i in range(q):
# #         SE(np.sqrt(np.e(y[i])), np.sqrt(np.e(PredictedlogRVforAll[i])), dates, function_method=label)Daily_Vol_df = daily_vol_combined
#
#
# MSE_daily,QL_daliy = VAR_MSE_QL(q=9, p=6, t=daily_warmup, n=len(LogRV_df[0])-daily_warmup,indicator="Daily")
# MSE_weekly,QL_weekly = VAR_MSE_QL(q=9, p=6, t=weekly_warmup, n=len(LogRV_df[1])-weekly_warmup,indicator="Weekly")
# MSE_monthly,QL_monthly = VAR_MSE_QL(q=9, p=6, t=monthly_warmup, n=len(LogRV_df[2])-monthly_warmup,indicator="Monthly")