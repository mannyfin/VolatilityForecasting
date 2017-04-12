import pandas as pd
import numpy as np


def knn_type(method_type='train', window_type='expanding window', vol_data=None, warmup=None, k=1):
    prediction = pd.Series()

    # "method 0 train "
    if method_type == 'train' and window_type =='expanding window':
        iterator = 0
        assert type(warmup) is int, "warmup-test case is not passed as an integer"
        assert type(vol_data) is pd.core.frame.DataFrame, "vol_data-train case is not passed as a DataFrame"
        observed = vol_data.iloc[warmup:]
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
            sigma = train_set.iloc[kn_index]
            # prediction = prediction.append(pd.Series([np.dot(alpha_j, sigma)], index=[iterator]))
            # the line below for passed DataFrames
            prediction = prediction.append(pd.Series([np.dot(alpha_j[alpha_j.keys()[0]], sigma[sigma.keys()[0]])],
                                                     index=[iterator]))
            iterator += 1

        return observed, prediction
    # "method 0 test "
    if method_type == 'test' and window_type =='expanding window':
        # iterator starts at 1 because of indexing vol_data[0:iterator] would give an error if iterator == 0
        iterator = 1
        assert type(warmup) is pd.core.frame.DataFrame, "warmup-test case is not passed as a DataFrame"
        assert type(vol_data) is pd.core.frame.DataFrame, "vol_data-test case is not passed as a DataFrame"
        # under testing, the observation data is just the vol_data

        observed = vol_data
        while iterator < len(vol_data)+1:
            # use the below for testing
            # while iterator < len(vol_data):
            # load in the datapoints
            # growing window
            # train_set = vol_data[0:(warmup + iterator)]
            test_set = pd.concat([warmup, vol_data[0:iterator]])
            # moving window
            # train_set = vol_data[iterator:(warmup+iterator)]

            last_sample = test_set.iloc[- 1]
            diff = last_sample - test_set
            # absdiff = diff.abs().sort_values()
            absdiff = diff.abs().sort_values(by=diff.keys()[0])
            kn_index = diff.reindex(absdiff.index)[1:k + 1].index + 1
            squared = absdiff[1:k + 1] ** 2
            c = 1 / np.sum(1 / squared)
            alpha_j = c / squared
            # sigma = vol_data[kn_index]
            sigma = test_set.iloc[kn_index]

            # prediction = prediction.append(pd.Series([np.dot(alpha_j, sigma)], index=[iterator]))

            # the line below for passed DataFrames
            # we minus 1 from the iterator here so that the index is still at zero...
            prediction = prediction.append(pd.Series([np.dot(alpha_j[alpha_j.keys()[0]], sigma[sigma.keys()[0]])],
                                                     index=[iterator-1]))
            iterator += 1
        return observed, prediction
    #     "method 1 train "

    if method_type == 'train' and window_type == 'time component':
        iterator = 0
        predict_multiple_m = []
        observed = vol_data.iloc[warmup:]
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

            # TODO standardize time
            time_comp = np.array(((last_sample.name - kn_index) / last_sample.name) ** 2)



            squared = np.array(absdiff[1:k + 1] ** 2).reshape(time_comp.shape) + time_comp
            c = 1 / np.sum(1 / squared)
            alpha_j = c / squared
            # sigma = vol_data[kn_index]
            sigma = vol_data.iloc[kn_index]

            # may need to add  .astype('float64') to the sigma below
            dt = np.sqrt(np.array(sigma ** 2).reshape(time_comp.shape) + time_comp**2)

            prediction = prediction.append(pd.Series([np.dot(alpha_j, dt)], index=[iterator]))
            # the line below for passed DataFrames
            # prediction = prediction.append(pd.Series([np.dot(alpha_j[alpha_j.keys()[0]], sigma[sigma.keys()[0]])],
            #                                          index=[iterator]))
            iterator += 1

        return observed, prediction

    # "method 1 test "
    if method_type == 'test' and window_type == 'time component':
        # iterator starts at 1 because of indexing vol_data[0:iterator] would give an error if iterator == 0
        iterator = 1
        assert type(warmup) is pd.core.frame.DataFrame, "warmup-test case is not passed as a DataFrame"
        assert type(vol_data) is pd.core.frame.DataFrame, "vol_data-test case is not passed as a DataFrame"
        # under testing, the observation data is just the vol_data

        observed = vol_data
        while iterator < len(vol_data) + 1:
            # use the below for testing
            # while iterator < len(vol_data):
            # load in the datapoints
            # growing window
            # train_set = vol_data[0:(warmup + iterator)]
            test_set = pd.concat([warmup, vol_data[0:iterator]])
            # moving window
            # train_set = vol_data[iterator:(warmup+iterator)]

            last_sample = test_set.iloc[- 1]
            diff = last_sample - test_set
            # absdiff = diff.abs().sort_values()
            absdiff = diff.abs().sort_values(by=diff.keys()[0])
            kn_index = diff.reindex(absdiff.index)[1:k + 1].index + 1

            # TODO standardize time
            # normalize time
            time_comp = np.array(((last_sample.name - kn_index) / last_sample.name) ** 2)
            # standardize time
            time_comp = (time_comp - np.mean(time_comp)) / np.std(time_comp)

            squared = np.array(absdiff[1:k + 1] ** 2).reshape(time_comp.shape) + time_comp
            c = 1 / np.sum(1 / squared)
            alpha_j = c / squared
            # sigma = vol_data[kn_index]
            sigma = test_set.iloc[kn_index]

            dt = np.sqrt(np.array(sigma.astype('float64') ** 2).reshape(time_comp.shape) + time_comp ** 2)

            prediction = prediction.append(pd.Series([np.dot(alpha_j, dt)], index=[iterator]))
            iterator += 1
        return observed, prediction

    # method 2 train
    if method_type == 'train' and window_type == 'spread component':
        iterator = 0
        predict_multiple_m = []
        observed = vol_data.iloc[warmup:]
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

            # TODO standardize time
            time_comp = np.array(((last_sample.name - kn_index) / last_sample.name) ** 2)



            squared = np.array(absdiff[1:k + 1] ** 2).reshape(time_comp.shape) + time_comp
            c = 1 / np.sum(1 / squared)
            alpha_j = c / squared
            # sigma = vol_data[kn_index]
            sigma = vol_data.iloc[kn_index]

            # may need to add  .astype('float64') to the sigma below
            dt = np.sqrt(np.array(sigma ** 2).reshape(time_comp.shape) + time_comp**2)

            prediction = prediction.append(pd.Series([np.dot(alpha_j, dt)], index=[iterator]))
            # the line below for passed DataFrames
            # prediction = prediction.append(pd.Series([np.dot(alpha_j[alpha_j.keys()[0]], sigma[sigma.keys()[0]])],
            #                                          index=[iterator]))
            iterator += 1

        return observed, prediction
    # method 2 test
    if method_type == 'test' and window_type == 'spread component':
        # iterator starts at 1 because of indexing vol_data[0:iterator] would give an error if iterator == 0
        iterator = 1
        assert type(warmup) is pd.core.frame.DataFrame, "warmup-test case is not passed as a DataFrame"
        assert type(vol_data) is pd.core.frame.DataFrame, "vol_data-test case is not passed as a DataFrame"
        # under testing, the observation data is just the vol_data

        observed = vol_data
        while iterator < len(vol_data) + 1:
            # use the below for testing
            # while iterator < len(vol_data):
            # load in the datapoints
            # growing window
            # train_set = vol_data[0:(warmup + iterator)]
            test_set = pd.concat([warmup, vol_data[0:iterator]])
            # moving window
            # train_set = vol_data[iterator:(warmup+iterator)]

            last_sample = test_set.iloc[- 1]
            diff = last_sample - test_set
            # absdiff = diff.abs().sort_values()
            absdiff = diff.abs().sort_values(by=diff.keys()[0])
            kn_index = diff.reindex(absdiff.index)[1:k + 1].index + 1

            # TODO standardize time
            # normalize time
            time_comp = np.array(((last_sample.name - kn_index) / last_sample.name) ** 2)
            # standardize time
            time_comp = (time_comp - np.mean(time_comp)) / np.std(time_comp)

            squared = np.array(absdiff[1:k + 1] ** 2).reshape(time_comp.shape) + time_comp
            c = 1 / np.sum(1 / squared)
            alpha_j = c / squared
            # sigma = vol_data[kn_index]
            sigma = test_set.iloc[kn_index]

            dt = np.sqrt(np.array(sigma.astype('float64') ** 2).reshape(time_comp.shape) + time_comp ** 2)

            prediction = prediction.append(pd.Series([np.dot(alpha_j, dt)], index=[iterator]))
            iterator += 1
        return observed, prediction
