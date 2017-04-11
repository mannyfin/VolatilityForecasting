import numpy as np
import pandas as pd
from Performance_Measure import *
from SEplot import se_plot as SE
from compatibility_check import compatibility_check as cc
import matplotlib.pyplot as plt
from KNN_method_type import knn_type
from optionsdict import *
import prediction_output

def KNN(method_number,vol_data, dates=None, k=1, warmup=100, filename=None, Timedt=None):

    vol_data_input = vol_data
    try:
        "support capability of dates passed together with vol_data and dates=None"
        dates = vol_data['Date']
        vol_data_input = vol_data['Volatility_Time']

    except KeyError:
        pass

    return KNNcalc(method_number=method_number,vol_data=vol_data_input, dates=dates, k=k, warmup=warmup,
                   filename=filename, Timedt=Timedt)


def KNNcalc(method_number, vol_data, dates=None, k=1, warmup=400, filename=None, Timedt=None):

    """
    # we now want to predict the volatility at time, t_warmup,
    # so we subtract vol @t_warmup from every point in train set
    # now we sort the abs value of these differences in ascending order (but without necessarily applying abs)
    # the first index is just the last_sample, so we will ignore it and choose the first 1-k samples' indices
    # recall in python indexing that the last index is not included, so for k = 1, only index 1 is chosen,
    # we add 1 to get the indices for the predicted vol
    # we now attempt to find the value of c. Below is ||sigma_t - sigma_k's||^2
    So how do we solve for c?
    Well we know that alpha_j = c / ||sigma_t - sigma_k's||
    and sum(alpha_j for all j's) = 1 = alpha_1 + alpha_2 + ... + alpha_k
    then:
    1 = c / ||sigma_t - sigma_1nn|| + c / ||sigma_t - sigma_2nn|| + ... c / ||sigma_t - sigma_knn||
    factoring out c:
    1 = c ( 1/||sigma_t - sigma_1nn|| + 1 / ||sigma_t - sigma_2nn|| + ... 1 / ||sigma_t - sigma_knn|| )
    So c = 1 / ( 1/||sigma_t - sigma_1nn|| + 1 / ||sigma_t - sigma_2nn|| + ... 1 / ||sigma_t - sigma_knn|| )
    # the sigmass are for the time point ahead of the nearest neighbor...these will be used for the prediction
    :param vol_data: pd.Series or pd.DataFrame object. If DF, then call a particular column using df.column_name
    :param k: nearest neighbors. Should be type(int)
    :param warmup: initial training period. Should be type(int)
    :param filename: None, for default case
    :return:
    
    
    method 0; growing window, skip first 100 data		
            -plot of mse vs k
            -table of mse vs k, give excel file or csv
            -all currencies on one ln(SE) plot
    method 1; rolling fixed window. 100 window size
            -optimize window size. 100 200 300, 400
            -plot mse vs k
            -table of mse vs k
    method 2: do it. Standardize t. Try standardizing sigma
            -plot of mse vs k
            -table of mse vs k
    method 3: dont do it

    """

    if filename is None:
        filename = " "
    # initialize
    prediction = pd.Series()
    iterator = 0
    try:
        # for training, warmup is just an integer, for testing warmup is a df
        # if method_number == '0-train':
        observed, prediction = knn_type(method_type=optionsdict(method_number)[0], window_type=optionsdict(method_number)[1],
                                  vol_data=vol_data, warmup=warmup, k=k)

        # elif method_number == '0-test':
        #     prediction = knn_type(method_type='test', window_type='expanding window', vol_data=vol_data,
        #                           warmup=warmup, k=k)

        # while iterator < (len(vol_data)-warmup):
        # # use the below for testing
        # # while iterator < len(vol_data):
        #     # load in the datapoints
        #     # growing window
        #     train_set = vol_data[0:(warmup + iterator)]
        #
        #
        #     # moving window
        #     # train_set = vol_data[iterator:(warmup+iterator)]
        #
        #     last_sample = train_set.iloc[- 1]
        #     diff = last_sample - train_set
        #     # absdiff = diff.abs().sort_values()
        #     absdiff = diff.abs().sort_values(by=diff.keys()[0])
        #     kn_index = diff.reindex(absdiff.index)[1:k + 1].index + 1
        #
        #     squared = absdiff[1:k + 1] ** 2
        #
        #     c = 1 / np.sum(1/squared)
        #     alpha_j = c/squared
        #     # sigma = vol_data[kn_index]
        #     sigma = vol_data.iloc[kn_index]
        #     # prediction = prediction.append(pd.Series([np.dot(alpha_j, sigma)], index=[iterator]))
        #     # the line below for passed DataFrames
        #     prediction = prediction.append(pd.Series([np.dot(alpha_j[alpha_j.keys()[0]],sigma[sigma.keys()[0]])],
        #                                              index=[iterator]))
        #     iterator += 1
            # print(prediction)
        # print(prediction)
    except:
        TypeError('Not a pd.Series or pd.DataFrame')
        ValueError("bad values")


    prediction = cc(observed, prediction)
    # now calculate MSE, QL and so forth
    Performance_ = PerformanceMeasure()
    # MSE = Performance_.mean_se(observed=vol_data.iloc[warmup:], prediction=prediction).transpose()
    MSE = Performance_.mean_se(observed=observed, prediction=prediction).transpose()
    # MSE.rename(index={0: 'MSE'})
    MSE.columns = ['MSE']

    QL = Performance_.quasi_likelihood(observed=observed, prediction=prediction).transpose()
    QL.columns = ['QL']
    label = str(filename) + " " + str(Timedt) + " ln(SE) (" + str(k) + ") KNN Volatility"
    print(label)
    """ return a plot of the Squared error"""

    if type(warmup)==int:
        # training case
        SE(observed, prediction, dates.iloc[warmup:], function_method=label)
    else:
        SE(observed, prediction, dates, function_method=label)

    plttitle = str(filename) + ' k = ' + str(k)
    plt.title(str(filename) + ' k = ' + str(k))
    plt.savefig(plttitle + '.png')
    plt.close()
    tempfix = pd.concat([MSE, QL], axis=1)

    if 'test' in method_number:
        prediction_output(prediction)


    # return MSE, QL
    return tempfix
