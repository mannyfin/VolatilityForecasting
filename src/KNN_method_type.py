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
            sigma = vol_data.iloc[kn_index]
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
            train_set = pd.concat([warmup, vol_data[0:iterator]])
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
            time_comp = np.array(((last_sample.name - kn_index) / last_sample.name) ** 2)

            squared = np.array(absdiff[1:k + 1] ** 2).reshape(time_comp.shape) + time_comp
            c = 1 / np.sum(1 / squared)
            alpha_j = c / squared
            # sigma = vol_data[kn_index]
            sigma = vol_data.iloc[kn_index]

            dt = np.sqrt(np.array(sigma ** 2).reshape(time_comp.shape) + time_comp)

            prediction = prediction.append(pd.Series([np.dot(alpha_j, dt)], index=[iterator]))
            # the line below for passed DataFrames
            # prediction = prediction.append(pd.Series([np.dot(alpha_j[alpha_j.keys()[0]], sigma[sigma.keys()[0]])],
            #                                          index=[iterator]))
            iterator += 1

        return observed, prediction















        #     train_set = vol_data[vol_data.keys()[0]][0:(warmup + iterator)].astype('float64')
        #     # while iterator < (len(vol_data) - warmup):
        #     # m = np.linspace(0, 20, 20)
        #     # diff = (last_sample - train_set) ** 2 + m * (train_set.index - train_set.index[-1]) ** 2
        #     last_sample = train_set.iloc[- 1].astype('float64')
        #     # test a whole bunch of m's
        #     diff = [(last_sample - train_set) ** 2 + m_iter * ((train_set.index - train_set.index[-1]) / (len(train_set))) ** 2
        #             for m_iter in m]
        #
        #     # normalize...
        #     # diff = (last_sample - train_set) - m * (train_set.index - train_set.index[-1])/len(train_set)
        #
        #     # absdiff = diff.abs().sort_values()
        #     # use the line below when testing multple m's. otherwise use the line above
        #     absdiff = [diff[i].abs().sort_values() for i in range(0, len(diff))]
        #
        #     # kn_index = diff.reindex(absdiff.index)[1:k + 1].index + 1
        #     # use line below for multiple m's
        #     kn_index = [diff[i].reindex(absdiff[i].index)[1:k + 1].index + 1 for i in range(0, len(diff))]
        #
        #     # squared = absdiff[1:k + 1]
        #     # use line below for multiple m's
        #     squared = [absdiff[i][1:k + 1] for i in range(0, len(diff))]
        #
        #     # c = 1 / sum(1 / squared)
        #     # alpha_j = c / squared
        #     # use two lines below for multiple m's
        #     c = [1 / sum(1 / squared[i]) for i in range(0, len(diff))]
        #     alpha_j = [c[i] / squared[i] for i in range(0, len(diff))]
        #
        #     # ds = np.sqrt(vol_data[kn_index].astype('float64') ** 2 + m * kn_index ** 2)
        #
        #     # ds = [np.sqrt(vol_data[kn_index[i][0]].astype('float64') ** 2 + (m[i] / (len(train_set) ** 2)) * (
        #     #         kn_index[i][0] / (len(train_set) ** 2)) ** 2) for i in range(0, len(diff))]
        #     ds = [np.sqrt(vol_data[kn_index[i][0]:kn_index[1][0]+1].astype('float64') ** 2 + (m[i] / (len(train_set) ** 2)) * (
        #             kn_index[i][0] / (len(train_set) ** 2)) ** 2) for i in range(0, len(diff))]
        #
        #     # ds = vol_data[kn_index].astype('float64') ** 2 + m * kn_index ** 2
        #     # ds = (vol_data[kn_index].astype('float64') + m * kn_index)/1000
        #
        #     # prediction = [prediction.append(pd.Series([np.dot(alpha_j[i], ds[i])], index=[iterator])) for i in range(0,len(diff))]
        #     # use line below for multiple m's
        #     predict_multiple_m.append([np.dot(alpha_j[i], ds[i]) for i in range(0, len(diff))])
        #
        # return observed, predict_multiple_m


    if method_type == 'train' and window_type == 'time component':
        iterator = 0
        predict_multiple_m = []
        observed = vol_data.iloc[warmup:]
        while iterator < (len(vol_data) - warmup):

            train_set = vol_data[vol_data.keys()[0]][0:(warmup + iterator)].astype('float64')
            # while iterator < (len(vol_data) - warmup):
            m = np.linspace(0, 20, 20)
            # diff = (last_sample - train_set) ** 2 + m * (train_set.index - train_set.index[-1]) ** 2
            last_sample = train_set.iloc[- 1].astype('float64')
            # test a whole bunch of m's
            diff = [(last_sample - train_set) ** 2 + m_iter * ((train_set.index - train_set.index[-1]) / (len(train_set))) ** 2
                    for m_iter in m]

            # normalize...
            # diff = (last_sample - train_set) - m * (train_set.index - train_set.index[-1])/len(train_set)

            # absdiff = diff.abs().sort_values()
            # use the line below when testing multple m's. otherwise use the line above
            absdiff = [diff[i].abs().sort_values() for i in range(0, len(diff))]

            # kn_index = diff.reindex(absdiff.index)[1:k + 1].index + 1
            # use line below for multiple m's
            kn_index = [diff[i].reindex(absdiff[i].index)[1:k + 1].index + 1 for i in range(0, len(diff))]

            # squared = absdiff[1:k + 1]
            # use line below for multiple m's
            squared = [absdiff[i][1:k + 1] for i in range(0, len(diff))]

            # c = 1 / sum(1 / squared)
            # alpha_j = c / squared
            # use two lines below for multiple m's
            c = [1 / sum(1 / squared[i]) for i in range(0, len(diff))]
            alpha_j = [c[i] / squared[i] for i in range(0, len(diff))]

            # ds = np.sqrt(vol_data[kn_index].astype('float64') ** 2 + m * kn_index ** 2)

            # ds = [np.sqrt(vol_data[kn_index[i][0]].astype('float64') ** 2 + (m[i] / (len(train_set) ** 2)) * (
            #         kn_index[i][0] / (len(train_set) ** 2)) ** 2) for i in range(0, len(diff))]
            ds = [np.sqrt(vol_data[kn_index[i][0]:kn_index[1][0]+1].astype('float64') ** 2 + (m[i] / (len(train_set) ** 2)) * (
                    kn_index[i][0] / (len(train_set) ** 2)) ** 2) for i in range(0, len(diff))]

            # ds = vol_data[kn_index].astype('float64') ** 2 + m * kn_index ** 2
            # ds = (vol_data[kn_index].astype('float64') + m * kn_index)/1000

            # prediction = [prediction.append(pd.Series([np.dot(alpha_j[i], ds[i])], index=[iterator])) for i in range(0,len(diff))]
            # use line below for multiple m's
            predict_multiple_m.append([np.dot(alpha_j[i], ds[i]) for i in range(0, len(diff))])

        return observed, predict_multiple_m

    # "method 1 test "
    if method_type == 'test' and window_type == 'time component':
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