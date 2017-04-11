import numpy as np
from sklearn.linear_model import LogisticRegression as Logit
from Performance_Measure import *
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

"""
this Python profile implements Logistic Regression for Forecaster 1, 2, 5 and 6

"""

def Predict_y_delta_star_training(train_sample, forecaster):
    """
    :param train_sample: could be train_sample_daily or train_sample_weekly (obtained in main)
    :param forecaster: forecaster could take values 1, 2, 5 or 6
    :return: the predicted y label, delta_star and the fitted model in the training sample
    """
    Model = Logit()
    if forecaster == 1:
        fittedModel = Model.fit(np.array(train_sample.vol_now).reshape(len(train_sample.vol_now),1),
                                np.array(train_sample.label))
        predicted_y_train = fittedModel.predict(np.array(train_sample.vol_now).reshape(len(train_sample.vol_now),1))
    elif forecaster == 2:
        inputdf = train_sample[['vol_now', 'ret_now']]
        fittedModel = Model.fit(np.array(inputdf).reshape(len(inputdf),2),
                                np.array(train_sample.label))
        predicted_y_train = fittedModel.predict(np.array(inputdf).reshape(len(inputdf),2))
    elif forecaster == 5:
        inputdf = train_sample[['vol_now','ret_now', 'vol_sqr_now']]
        fittedModel = Model.fit(np.array(inputdf).reshape(len(inputdf),3),
                                np.array(train_sample.label))
        predicted_y_train = fittedModel.predict(np.array(inputdf).reshape(len(inputdf),3))
    elif forecaster == 6:
        inputdf = train_sample[['vol_now',  'ret_now','volxret_now','vol_sqr_now']]
        fittedModel = Model.fit(np.array(inputdf).reshape(len(inputdf),4),
                                np.array(train_sample.label))
        predicted_y_train = fittedModel.predict(np.array(inputdf).reshape(len(inputdf),4))

    vol_today = np.array(train_sample.vol_now).reshape(len(train_sample.vol_now),1)
    vol_tmr = np.array(train_sample.vol_future).reshape(len(train_sample.vol_future),1)
    # element wise multiplication of vol_today and predicted_y_train to get P
    P = np.array([a * b for a, b in zip(vol_today, predicted_y_train)])
    # implement matrix multiplication by using np.dot
    delta_star = np.dot( np.transpose(P), (vol_tmr - vol_today) ) / np.dot( np.transpose(P), P )


    return predicted_y_train, delta_star.flatten()[0], fittedModel


def test_performance_LR(train_sample,test_sample,forecaster):
    """
    :param train_sample: could be train_sample_daily or train_sample_weekly (obtained in main)
    :param test_sample: could be test_sample_daily or test_sample_weekly (obtained in main)
    :param forecaster: forecaster could take values 1, 2, 5 or 6
    :return: MSE, QL, SE plot, observed_test_sample_vol_future, predicted_test_sample_vol_future
    """
    output = Predict_y_delta_star_training(train_sample, forecaster=forecaster)
    fittedModel = output[2]
    delta_star = output[1]
    if forecaster == 1:
        predicted_y_test =  fittedModel.predict(np.array(test_sample.vol_now).reshape(len(test_sample.vol_now),1))
    elif forecaster == 2:
        inputdf = test_sample[['vol_now', 'ret_now']]
        predicted_y_test = fittedModel.predict(np.array(inputdf).reshape(len(inputdf),2))
    elif forecaster == 5:
        inputdf = test_sample[['vol_now','ret_now', 'vol_sqr_now']]
        predicted_y_test = fittedModel.predict(np.array(inputdf).reshape(len(inputdf),3))
    elif forecaster == 6:
        inputdf = test_sample[['vol_now',  'ret_now','volxret_now','vol_sqr_now']]
        predicted_y_test = fittedModel.predict(np.array(inputdf).reshape(len(inputdf),4))

    element1 = np.array(test_sample.vol_now)
    element2 = 1 + delta_star * predicted_y_test
    predicted_test_sample_vol_future = pd.Series([a * b for a, b in zip(element1, element2)])
    observed_test_sample_vol_future = pd.Series(test_sample.vol_future)

    # measuring prediction performance in the test sample
    Performance_ = PerformanceMeasure()
    MSE = Performance_.mean_se(observed_test_sample_vol_future,predicted_test_sample_vol_future)
    QL = Performance_.quasi_likelihood(observed_test_sample_vol_future.astype('float64'),
                                       predicted_test_sample_vol_future.astype('float64'))
    SE = [(observed_test_sample_vol_future[i] - predicted_test_sample_vol_future[i])**2 for i in range(len(observed_test_sample_vol_future))]
    ln_SE = pd.Series(np.log(SE))
    return MSE, QL, ln_SE


