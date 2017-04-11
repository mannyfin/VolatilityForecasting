import pandas as pd
import numpy as np


def knn_type(method_type='train', window_type='expanding window', vol_data=None, warmup=None, k=1):
    prediction = pd.Series()
    if method_type == 'train' and window_type =='expanding window':
        iterator = 0
        assert type(warmup) is int, "warmup is not passed as an integer"

        while iterator < (len(vol_data) - warmup):
            # use the below for testing
            # while iterator < len(vol_data):
            # load in the datapoints
            # growing window
            train_set = vol_data[0:(warmup + iterator)]

            # moving window
            # train_set = vol_data[iterator:(warmup+iterator)]

            last_sample = train_set.iloc[- 1]
            diff = last_sample - train_set
            # absdiff = diff.abs().sort_values()
            absdiff = diff.abs().sort_values(by=diff.keys()[0])
            kn_index = diff.reindex(absdiff.index)[1:k + 1].index + 1
            squared = absdiff[1:k + 1] ** 2
            c = 1 / np.sum(1 / squared)
            alpha_j = c / squared
            # sigma = vol_data[kn_index]
            sigma = vol_data.iloc[kn_index]
            # prediction = prediction.append(pd.Series([np.dot(alpha_j, sigma)], index=[iterator]))
            # the line below for passed DataFrames
            prediction = prediction.append(pd.Series([np.dot(alpha_j[alpha_j.keys()[0]], sigma[sigma.keys()[0]])],
                                                     index=[iterator]))
            iterator += 1
        return prediction

    elif method_type == 'train' and window_type == 'moving fixed window':
        iterator = 0
        # while iterator < (len(vol_data) - warmup):
            # use the below for testing
        while iterator < len(vol_data):
            # load in the datapoints
            # # growing window
            # train_set = vol_data[0:(warmup + iterator)]
            # moving window
            test_set = vol_data[iterator:(warmup+iterator)]

            last_sample = test_set.iloc[- 1]
            diff = last_sample - test_set
            # absdiff = diff.abs().sort_values()
            absdiff = diff.abs().sort_values(by=diff.keys()[0])
            kn_index = diff.reindex(absdiff.index)[1:k + 1].index + 1
            squared = absdiff[1:k + 1] ** 2
            c = 1 / np.sum(1 / squared)
            alpha_j = c / squared
            # sigma = vol_data[kn_index]
            sigma = vol_data.iloc[kn_index]
            # prediction = prediction.append(pd.Series([np.dot(alpha_j, sigma)], index=[iterator]))
            # the line below for passed DataFrames
            prediction = prediction.append(pd.Series([np.dot(alpha_j[alpha_j.keys()[0]], sigma[sigma.keys()[0]])],
                                                     index=[iterator]))
            iterator += 1
        return prediction