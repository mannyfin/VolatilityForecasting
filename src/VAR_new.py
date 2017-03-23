# This file implements VAR (vector auto regression) described  in the paper
# Wilms, Ines, Jeroen Rombouts, and Christophe Croux. "Lasso-based forecast combinations for forecasting realized variances." (2016).

# We only implements VAR on daily volatility data.
# We take log of annualized volatility to get logRV.
# We divide the data into training sample (2/3) and test sample (1/3).
# For lag p = 1, look-back rolling window is 100.
# For lag p = 2, look-back rolling window is 200.
# For lag p = 3, look-back rolling window is 300.
# From the training sample, we get the best p giving the smallest MSE;
# Using the optimal p, we fit the model and make predictions on the test sample.
# We used a rolling window method rather than a growing window method.


import numpy as np
import pandas as pd
from Performance_Measure import *
from sklearn.linear_model import LinearRegression as lr

'''
    Construct y as a list of 9 lists.
    The k-th list inside y is a series of logRV for the k-th currency pair for k=1,2,...,9
'''
def get_y(LogRV_df,q, t,n):
# def get_y(LogRV_df, q, p, t, n):

    '''
    :param LogRV_df: LogRV_df = np.log(daily_vol_combined)
    :param q: q=9 in this project since we have 9 currency pairs
    :param p: p is lag
    :param t: t = indicator
    :param n: n = length of the look-back rolling window
    :return: y as inputs into LR for all currency pairs
    '''
    y = []
    # len(LogRV_df.keys()) is q is 9
    for i in range(q):
        # y_i= []
        # for k in range(n):
            # "t here is bound to predictLogRV...."
        # y_i.append( LogRV_df.iloc[t+k][i] )
        # y.append(y_i)
        y.append(LogRV_df.iloc[t:t+n, i])

    return y


def x_mat_t_n_qp(LogRV_df,q, p, t,n):
    '''
    :param LogRV_df: LogRV_df = np.log(daily_vol_combined)
    :param q: q=9 in this project since we have 9 currency pairs
    :param p: p is lag
    :param t: t = time indicator
    :param n: n = length of the look-back rolling window
    :return: the x matrix as a input for regression, where the dimension of x is n*(qp)
    '''
    x =  pd.DataFrame()
    for m in range(n):
        x_t_vec = []
        # this is different from the k in the y creation
        for k in range(q):
            for i in range(1, p+1):
                "t here is bound to predictLogRV...."
                # Why not just call these in the right order and then reverse them?
                # Right now they are called as: element [t+m-i] = 3, 2, 1 for iteration i = 1
                x_t_vec.append(LogRV_df.iloc[t+m-i][k])
        x = x.append([x_t_vec])
    return x

'''
     Fitting parameters and making prediction based on fitted models
     PredictedlogRV collects the predicted logRV for all 9 currency pairs

'''
# def predictlogRV(LogRV_df,q,p,t,n):


def predictlogRV(LogRV_df, q, p, n,stringin=None):

    '''
    :param LogRV_df: LogRV_df = np.log(daily_vol_combined)
    :param q: q=9 in this project since we have 9 currency pairs
    :param p: p is lag
    :param n: n = length of the look-back rolling window
    :return: the predicted logRV for all 9 currency pairs in the training sample
    '''
    # diff range of t
    PredictedlogRVforAll = []
    obs_yforAll = []
    for t in range(p,int(2/3*len(LogRV_df))-n):
        x = x_mat_t_n_qp(LogRV_df,q, p, t,n)
        y = get_y(LogRV_df,q, t,n)
        PredictedlogRV = []
        obs_y = []
        for i in range(q):
            A = lr()
            A.fit( x, y[i] )
            b = A.coef_
            c = A.intercept_
            x_used_in_pred = x_mat_t_n_qp(LogRV_df,q, p, t+1,n)
            PredictedlogRV.append( A.predict(x_used_in_pred.tail(1).values.reshape(1, -1))[0])
            obs_y.append(get_y(LogRV_df, q, t + 1, n)[i][-1])
        PredictedlogRVforAll.append(PredictedlogRV)
        obs_yforAll.append(obs_y)
        print(str(stringin) + str(p) + "_t_" + str(t))

        mean_MSE, mean_QL, MSEforAll, QLforAll = VAR_MSE_QL(PredictedlogRVforAll, obs_yforAll, q)

    return mean_MSE, mean_QL, MSEforAll, QLforAll

'''
    Obtaining MSE and QL
'''
def VAR_MSE_QL(PredictedlogRVforAll, y, q):
    '''
    :param LogRV_df: LogRV_df = np.log(daily_vol_combined)
    :param y: realized logRV for all currency pairs
    :param q: q=9 in this project since we have 9 currency pairs
    :param p: p is lag
    :param n: n = length of the look-back rolling window
    :param TrainOrTest: TrainOrTest = "Train" or "Test"
    :return: MSE, QL and SE plot
    '''
    # if TrainOrTest == "Train":
    #     PredictedlogRVforAll, y = predictlogRV(LogRV_df,q, p, n, 'train_p_')[0:2]
    #
    # elif TrainOrTest == "Test":
    #     PredictedlogRVforAll, y = predictlogRV(LogRV_df, q, p, n, 'test_')[0:2]

    y = np.array(y).T
    PredictedlogRVforAll = np.array(PredictedlogRVforAll).T

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

    return mean_MSE, mean_QL, MSEforAll, QLforAll

'''
    Obtaining optimal p
'''

def optimal_p(LogRV_df,q,p_series):
    '''

    :param LogRV_df: LogRV_df = np.log(daily_vol_combined)
    :param q: q = 9  # q is the number of currency pairs
    :param p_series: p_series = [1, 2, 3]  # p is lag, picking 1,2 and 3 according to Amin's suggesting
    :return: optimal_p out of 1,2,3
    '''
    VAR_p1= predictlogRV(LogRV_df, q, p=p_series[0], n=p_series[0]*100, stringin="Train")
    VAR_p2= predictlogRV(LogRV_df, q, p=p_series[1], n=p_series[1]*100, stringin="Train")
    VAR_p3= predictlogRV(LogRV_df, q, p=p_series[2], n=p_series[2]*100, stringin="Train")

    MSEs = [VAR_p1[0],VAR_p2[0],VAR_p3[0]]
    QLs = [VAR_p1[1],VAR_p2[1],VAR_p3[1]]

    optimal_p_MLE = MSEs.index(min(MSEs))+1 # the optimal p according to MLE criterion is 3
    optimal_p_QL = QLs.index(min(QLs))+1 # the optimal p according to QL criterion is 3 as well
    optimal_p = optimal_p_MLE # we use the optimal p according to MLE criterion as the optimal p
    return optimal_p


'''
    Obtaining squared errors
'''


def VAR_SE(LogRV_df, q, p_series, data):
# def VAR_SE(LogRV_df, q, p_series, daily_lookback_series, data):
    '''

    :param LogRV_df: LogRV_df = np.log(daily_vol_combined)
    :param q: q=9
    :param p_series: p_series=[1,2,3]
    :param data: used in plotting SE
    :return: SE plot
    '''
    # TODO: input correct data input
    # len_training = int(2 / 3 * len(LogRV_df))
    p = optimal_p(LogRV_df,q,p_series)
    # p = optimal_p(LogRV_df,q,p_series,daily_lookback_series,len_training)
    t= daily_lookback_series[p-1]
    n= p*100
    # should be test data passed as string
    PredictedlogRVforAll,y = predictlogRV(LogRV_df, q, p, n)[0:2]
    # PredictedlogRVforAll = predictlogRV(LogRV_df, q, p, t, n)[0]
    # y = predictlogRV_trainingSample(LogRV_df, q, p, n)[1]
    # y = get_y(LogRV_df, q, t, n)
    # y = get_y(LogRV_df, q, p, t, n)
    label = "VAR"
    # TODO: change this label
    for i in range(q):
        SE(np.sqrt(np.e(y[i])), np.sqrt(np.e(PredictedlogRVforAll[i])), dates, function_method=label)
        # TODO: get dates
        # TODO: add return

'''
    Using the optimal p on the test sample
'''
def Test_Sample_MSE_QL(LogRV_df,q,p_series):
# def Test_Sample_MSE_QL(LogRV_df,q,p_series,daily_warmup_series):
#     len_training = int(2 / 3 * len(LogRV_df))
    p = optimal_p(LogRV_df,q,p_series)
    # p = optimal_p(LogRV_df,q,p_series,daily_warmup_series,len_training)
    MSE_QL_optimal_p = predictlogRV(LogRV_df,q,p, n=p*100, stringin="Test")
    MSE_optimal_p_avg = MSE_QL_optimal_p[0] # average MSE of 9 currency pairs
    QL_optimal_p_avg = MSE_QL_optimal_p[1]  # average QL of 9 currency pairs
    MSE_optimal_p_forAll = MSE_QL_optimal_p[2] # MSE of all 9 currency pairs
    QL_optimal_p_forAll = MSE_QL_optimal_p[3]  # QL of all 9 currency pairs


    return MSE_QL_optimal_p,MSE_optimal_p_avg,QL_optimal_p_avg,MSE_optimal_p_forAll,QL_optimal_p_forAll

# Test_Sample_MSE_QL(LogRV_df = np.log(daily_vol_combined), q=9, p_series=[1,2,3])