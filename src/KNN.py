# Using KNN method on daily and weekly data

import numpy as np


def KNN(vol_data, k, warmup):
    '''
    :param vol_data: vol_data = vol_data['Volatility_Time']
    :param k: number of nearest neignbors to find
    :param warmup: 400 for daily data and 70 for weekly data
    :return: prediction
    '''
    # load in the first int(warmup) datapoints
    train_set = vol_data[:warmup]
    # TODO: rolling window
    # we now want to predict the volatility at time, t_warmup,
    # so we subtract vol @t_warmup from every point in train set
    last_sample = train_set[warmup - 1]
    diff = last_sample - train_set
    absdiff = diff.abs().sort_values()
    # now we sort the abs value of these differences in ascending order (but without necessarily applying abs)
    kn_index = diff.reindex(absdiff.index)[1:k + 1].index + 1

    # the first index is just the last_sample, so we will ignore it and choose the first 1-k samples' indices
    # recall in python indexing that the last index is not included, so for k = 1, only index 1 is chosen,
    # we add 1 to get the indices for the predicted vol

    # we now attempt to find the value of c. Below is ||sigma_t - sigma_k's||^2
    # are you sure the formula below is correct for 1D? I see it okay for >=2D, but idk about 1D. I would think to use
    # manhatten distance
    squared = absdiff[1:k + 1] ** 2

    """
    So how do we solve for c?
    Well we know that alpha_j = c / ||sigma_t - sigma_k's||

    and sum(alpha_j for all j's) = 1 = alpha_1 + alpha_2 + ... + alpha_k
    then:
    1 = c / ||sigma_t - sigma_1nn|| + c / ||sigma_t - sigma_2nn|| + ... c / ||sigma_t - sigma_knn||

    factoring out c:

    1 = c ( 1/||sigma_t - sigma_1nn|| + 1 / ||sigma_t - sigma_2nn|| + ... 1 / ||sigma_t - sigma_knn|| )

    So c = 1 / ( 1/||sigma_t - sigma_1nn|| + 1 / ||sigma_t - sigma_2nn|| + ... 1 / ||sigma_t - sigma_knn|| )
    """
    c = 1 / sum(1/squared)
    print("check sum of alphaj: " + sum(c/squared))
    print("Does this equal 1? ")
    alpha_j = c/squared
    # the sigmas below are for the time point ahead of the nearest neighbor...these will be used for the prediction
    sigma = vol_data[kn_index]
    predicted = np.dot(alpha_j, sigma)
    return predicted


def KNN_MSE_QL(vol_data, k, warmup):
    #TODO: add MSE and QL here. This function can be used for all dataset

def KNN_MSE_QL(vol_data, k, warmup, filename):
#   TODO: add MSE and QL here. This function calls KNN_MSE_QL(vol_data, k, warmup), but is designed for the files we have


def KNN_SE(vol_data, k, warmup, filename):
    # TODO: add SE here. This function calls KNN_MSE_QL(vol_data, k, warmup), but is designed for the files we have


# def KNN(vol_data, k, warmup, filename):
#     # load in the first int(warmup) datapoints
#     train_set = vol_data['Volatility_Time'][:warmup]
#     # we now want to predict the volatility at time, t_warmup,
#     # so we subtract vol @t_warmup from every point in train set
#     last_sample = train_set[warmup - 1]
#     diff = last_sample - train_set
#     absdiff = diff.abs().sort_values()
#     # now we sort the abs value of these differences in ascending order (but without necessarily applying abs)
#     kn_index = diff.reindex(absdiff.index)[1:k + 1].index + 1
#
#     # the first index is just the last_sample, so we will ignore it and choose the first 1-k samples' indices
#     # recall in python indexing that the last index is not included, so for k = 1, only index 1 is chosen,
#     # we add 1 to get the indices for the predicted vol
#
#     # we now attempt to find the value of c. Below is ||sigma_t - sigma_k's||^2
#     # are you sure the formula below is correct for 1D? I see it okay for >=2D, but idk about 1D. I would think to use
#     # manhatten distance
#     squared = absdiff[1:k + 1] ** 2
#
#     """
#     So how do we solve for c?
#     Well we know that alpha_j = c / ||sigma_t - sigma_k's||
#
#     and sum(alpha_j for all j's) = 1 = alpha_1 + alpha_2 + ... + alpha_k
#     then:
#     1 = c / ||sigma_t - sigma_1nn|| + c / ||sigma_t - sigma_2nn|| + ... c / ||sigma_t - sigma_knn||
#
#     factoring out c:
#
#     1 = c ( 1/||sigma_t - sigma_1nn|| + 1 / ||sigma_t - sigma_2nn|| + ... 1 / ||sigma_t - sigma_knn|| )
#
#     So c = 1 / ( 1/||sigma_t - sigma_1nn|| + 1 / ||sigma_t - sigma_2nn|| + ... 1 / ||sigma_t - sigma_knn|| )
#     """
#     c = 1 / sum(1/squared)
#     print("check sum of alphaj: " + sum(c/squared))
#     print("Does this equal 1? ")
#     alpha_j = c/squared
#     # the sigmas below are for the time point ahead of the nearest neighbor...these will be used for the prediction
#     sigma = vol_data.Volatility_Time[kn_index]
#     predicted = np.dot(alpha_j, sigma)
