import numpy as np
from Performance_Measure import *
from SEplot import se_plot as SE
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import table


# choose the method in line 10 under method=[number] for number = 0,1,2,3
def KNN(vol_data, k=1, warmup=100, filename=None, Timedt=None, method=[3]):
    vol_data_input = vol_data.iloc[:,1]
    dates = pd.Series(vol_data.Date)

    # This can be done more efficiently by moving k list directly into k
    #
    knns = [[ks, m, KNNcalc(vol_data=vol_data_input, dates =dates, k=ks, warmup=warmup,filename=filename, Timedt=Timedt, method=m)]
            for count, m in enumerate(method) for ks in np.linspace(1,20,20)]
    # ks=20
    # knns = [[ks, m, KNNcalc(vol_data=vol_data_input, dates =dates, k=ks, warmup=warmup,filename=filename, Timedt=Timedt, method=m)]
    #         for count, m in enumerate(method)]
    mse = [knns[i][2][0] for i in range(len(knns))]
    ql = [knns[i][2][1] for i in range(len(knns))]
    kval= [int(knns[i][0]) for i in range(len(knns))]
    one_method_result = pd.DataFrame(np.transpose([kval, mse, ql]), columns=['k', 'MSE', 'QL'])
    # one_method_result = one_method_result.set_index('k')
    one_method_result.plot('k', 'MSE', figsize=[12, 7]).set_title(filename)
    one_method_result.plot('k', 'QL', figsize=[12, 7]).set_title(filename)

    # making a table
    fig, ax = plt.subplots()  # set size frame
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis
    ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
    tabla = table(ax, one_method_result.round(7), loc='center',
                  colWidths=[0.2] * len(one_method_result.columns))  # where df is your data frame
    tabla.auto_set_font_size(False)  # Activate set fontsize manually
    tabla.set_fontsize(10)  # if ++fontsize is necessary ++colWidths
    tabla.scale(1, 1)
    # feel free to comment out the line below for additional speed. But may cause overflow errors
    # with too many figs produced
    # plt.show()

    return one_method_result #knns[-1][2]


def KNNcalc(vol_data, dates=None, k=1, warmup=100, filename=None, Timedt=None, method=1, m=0):
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
    """
    k = int(k)
    print(k)
    # k=20
    growing = True
    if method == 1: growing = False
    if method == 2: m = 1

    if filename is None:
        filename = " "
    # initialize
    prediction = pd.Series()
    predict_multiple_m =[]
    iterator = 0
    try:
        while iterator < (len(vol_data) - warmup):
            # load in the datapoints
            # moving window (now changed to growing)
            if not growing:
                train_set = vol_data[0:(warmup + iterator)]
            else:
                train_set = vol_data[iterator:warmup + iterator]
            last_sample = train_set.iloc[- 1]

            if method == 1 or method == 0:
                diff = last_sample - train_set
                absdiff = diff.abs().sort_values()
                kn_index = diff.reindex(absdiff.index)[1:k + 1].index + 1
                squared = absdiff[1:k + 1] ** 2

                c = 1 / sum(1 / squared)
                alpha_j = c / squared
                sigma = vol_data[kn_index]
                prediction = prediction.append(pd.Series([np.dot(alpha_j, sigma)], index=[iterator]))

            elif method == 2 or method == 3:
                m = np.linspace(0,20,20)
                # diff = (last_sample - train_set) ** 2 + m * (train_set.index - train_set.index[-1]) ** 2

                # test a whole bunch of m's
                diff = [(last_sample - train_set) ** 2 + m_iter * ((train_set.index - train_set.index[-1])/(len(train_set))) ** 2 for
                        m_iter in m]
                # normalize...
                # diff = (last_sample - train_set) - m * (train_set.index - train_set.index[-1])/len(train_set)

                # absdiff = diff.abs().sort_values()
                # use the line below when testing multple m's. otherwise use the line above
                absdiff = [diff[i].abs().sort_values() for i in range(0, len(diff))]

                # kn_index = diff.reindex(absdiff.index)[1:k + 1].index + 1
                # use line below for multiple m's
                kn_index = [diff[i].reindex(absdiff[i].index)[1:k + 1].index + 1 for i in range(0,len(diff))]

                # squared = absdiff[1:k + 1]
                # use line below for multiple m's
                squared = [absdiff[i][1:k + 1] for i in range(0,len(diff))]

                # c = 1 / sum(1 / squared)
                # alpha_j = c / squared
                #use two lines below for multiple m's
                c = [1 / sum(1 / squared[i]) for i in range(0,len(diff))]
                alpha_j = [c[i] / squared[i] for i in range(0,len(diff))]

                # ds = np.sqrt(vol_data[kn_index].astype('float64') ** 2 + m * kn_index ** 2)

                ds = [np.sqrt(vol_data[kn_index[i]].astype('float64') ** 2 + (m[i]/(len(train_set)**2)) * (kn_index[i]/(len(train_set)**2)) ** 2)
                      for i in range(0, len(diff))]

                # ds = vol_data[kn_index].astype('float64') ** 2 + m * kn_index ** 2
                # ds = (vol_data[kn_index].astype('float64') + m * kn_index)/1000

                # prediction = [prediction.append(pd.Series([np.dot(alpha_j[i], ds[i])], index=[iterator])) for i in range(0,len(diff))]
                # use line below for multiple m's
                predict_multiple_m.append([np.dot(alpha_j[i], ds[i]) for i in range(0, len(diff))])

            else:
                print("Unexpected method")


            iterator += 1
            # print(prediction)
            # print(prediction)
    except:
        TypeError('Not a pd.Series or pd.DataFrame')
        ValueError("bad values")

    # now calculate MSE, QL and so forth
    Performance_ = PerformanceMeasure()
    if len(m)==1:
        MSE = Performance_.mean_se(observed=vol_data.iloc[warmup:], prediction=prediction)
        QL = Performance_.quasi_likelihood(observed=vol_data.iloc[warmup:].astype('float64'), prediction=prediction)
        label = str(filename.replace(".csv", ""))  # + " " + str(Timedt) + " SE (" + str(k) + ") KNN Volatility"
        # print(label,MSE)
        """ return a plot of the Squared error"""
        SE(vol_data.iloc[warmup:], prediction, dates.iloc[warmup:], function_method=label)  # , mode="no log")
        plt.title(str(filename) + ' k = ' + str(k))
    elif len(m)>1:
        mse_app=[]
        for count, elem in enumerate(m):
            mcombined = [predict_multiple_m[i][count] for i in range(0, len(predict_multiple_m))]
            MSE = Performance_.mean_se(observed=vol_data.iloc[warmup:], prediction=mcombined)
            QL=1  #we dont really need QL atm
            mse_app.append(MSE)

        plt.plot(m,mse_app)
        plt.xlabel('m')
        plt.ylabel('MSE')
        plt.title(str(filename)+' time penalty')
        plt.show()
        # label = str(filename.replace(".csv", ""))  # + " " + str(Timedt) + " SE (" + str(k) + ") KNN Volatility"
        # # print(label,MSE)
        # """ return a plot of the Squared error"""
        # SE(vol_data.iloc[warmup:], prediction, dates.iloc[warmup:], function_method=label)  # , mode="no log")
        # plt.title(str(filename) + ' k = ' + str(k))
    return MSE, QL

    # sanity check:  len(train_set) + len(prediction) == len(vol_data)