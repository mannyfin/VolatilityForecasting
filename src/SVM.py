import numpy as np
from sklearn.linear_model import LogisticRegression as Logit
from Performance_Measure import *
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC as SVC


def SVM_Predict_y_delta_star_training(train_sample, forecaster, model ,C, p=None, q=None):
    """
    :param train_sample: could be train_sample_daily or train_sample_weekly (obtained in main)
    :param forecaster: forecaster could take values 1, 2, 3, 4 5 or 6
    :param model: "SVM","KernelSVM_rbf","KernelSVM_sigmoid"
    :param C: margin in the SVM model, needed to be optimized
    :param p: p is a parameter in forecaster 3 and forecaster 4 
    :param q: q is a parameter in forecaster 4    
    :return: the predicted y label, benchmark_delta_star (the benchmark delta_star) and the fitted model in the training sample
    """
    if model == "SVM":
        Model = SVC(C, kernel='linear')
    elif model == "KernelSVM_rbf":
        Model = SVC(C, kernel='rbf')
    elif model == "KernelSVM_sigmoid":
        Model = SVC(C, kernel='sigmoid')

    if forecaster == 1:
        train_sample_new = train_sample.copy()
        fittedModel = Model.fit(np.array(train_sample_new.vol_now).reshape(len(train_sample_new.vol_now),1),
                                np.array(train_sample_new.label))
        predicted_y_train = fittedModel.predict(np.array(train_sample_new.vol_now).reshape(len(train_sample_new.vol_now),1))
    elif forecaster == 2:
        train_sample_new = train_sample.copy()
        inputdf = train_sample_new[['vol_now', 'ret_now']]
        fittedModel = Model.fit(np.array(inputdf).reshape(len(inputdf),2),
                                np.array(train_sample_new.label))
        predicted_y_train = fittedModel.predict(np.array(inputdf).reshape(len(inputdf),2))
    elif forecaster == 3:
        train_sample_new = train_sample.copy()
        train_sample_new["E_p"] = train_sample_new["vol_now"].rolling(p).mean()
        train_sample_new.dropna(inplace=True, axis=0)

        fittedModel = Model.fit(np.array(train_sample_new.E_p).reshape(len(train_sample_new.E_p), 1),
                                np.array(train_sample_new.label))
        predicted_y_train = fittedModel.predict(np.array(train_sample_new.E_p).reshape(len(train_sample_new.E_p), 1))
    elif forecaster == 4:
        train_sample_new = train_sample.copy()
        train_sample_new["E_p"] = train_sample_new["vol_now"].rolling(p).mean()
        train_sample_new["E_q"] = train_sample_new["ret_now"].rolling(q).mean()
        train_sample_new.dropna(inplace=True, axis=0)

        inputdf = train_sample_new[['E_p', 'E_q']]
        fittedModel = Model.fit(np.array(inputdf).reshape(len(inputdf), 2),
                                np.array(train_sample_new.label))
        predicted_y_train = fittedModel.predict(np.array(inputdf).reshape(len(inputdf), 2))
    elif forecaster == 5:
        train_sample_new = train_sample.copy()
        inputdf = train_sample_new[['vol_now','ret_now', 'vol_sqr_now']]
        fittedModel = Model.fit(np.array(inputdf).reshape(len(inputdf),3),
                                np.array(train_sample_new.label))
        predicted_y_train = fittedModel.predict(np.array(inputdf).reshape(len(inputdf),3))
    elif forecaster == 6:
        train_sample_new = train_sample.copy()
        inputdf = train_sample_new[['vol_now',  'ret_now','volxret_now','vol_sqr_now']]
        fittedModel = Model.fit(np.array(inputdf).reshape(len(inputdf),4),
                                np.array(train_sample_new.label))
        predicted_y_train = fittedModel.predict(np.array(inputdf).reshape(len(inputdf),4))

    vol_today = np.array(train_sample_new.vol_now).reshape(len(train_sample_new.vol_now), 1)
    vol_tmr = np.array(train_sample_new.vol_future).reshape(len(train_sample_new.vol_future), 1)
    # element wise multiplication of vol_today and predicted_y_train to get P
    P = np.array([a * b for a, b in zip(vol_today, predicted_y_train)])
    # implement matrix multiplication by using np.dot
    benchmark_delta_star = np.dot(np.transpose(P), (vol_tmr - vol_today)) / np.dot(np.transpose(P), P)

    return predicted_y_train, benchmark_delta_star.flatten()[0], fittedModel, train_sample_new

def SVM_Validation_Training(train_sample, forecaster, model, C, p=None, q=None):
    """
    :param train_sample: could be train_sample_daily or train_sample_weekly (obtained in main)
    :param forecaster: forecaster could take values 1, 2, 3, 4 5 or 6
    :param model: "SVM","KernelSVM_rbf","KernelSVM_sigmoid"
    :param C: margin in the SVM model, needed to be optimized
    :param p: p is a parameter in forecaster 3 and forecaster 4 
    :param q: q is a parameter in forecaster 4    
    :return: MSE of one validation in the training sample where the entire training sample 
            is randomly divided into the training in sample set and the training out of sample set once
    """
    output = SVM_Predict_y_delta_star_training(train_sample, forecaster, model ,C, p, q)
    train_sample_new = output[3]
    benchmark_delta_star = output[1]

    index_train_in_sample = np.random.choice(len(train_sample_new), int(len(train_sample_new) * 4 / 5), replace=False)
    index_train_in_sample = np.sort(index_train_in_sample)
    index_train_out_of_sample = set(range(len(train_sample_new))) - set(index_train_in_sample)

    train_in_sample = train_sample_new.ix[index_train_in_sample]
    train_out_of_sample = train_sample_new.ix[index_train_out_of_sample]

    if model == "SVM":
        Model = SVC(C, kernel='linear')
    elif model == "KernelSVM_rbf":
        Model = SVC(C, kernel='rbf')
    elif model == "KernelSVM_sigmoid":
        Model = SVC(C, kernel='sigmoid')

    if forecaster == 1:
        fittedModel = Model.fit(np.array(train_in_sample.vol_now).reshape(len(train_in_sample.vol_now),1),
                                np.array(train_in_sample.label))
        predicted_y_train_out_of_sample =  fittedModel.predict(np.array(train_out_of_sample.vol_now).reshape(len(train_out_of_sample.vol_now),1))
    elif forecaster == 2:
        inputdf = train_in_sample[['vol_now', 'ret_now']]
        fittedModel = Model.fit(np.array(inputdf).reshape(len(inputdf),2),
                                np.array(train_in_sample.label))
        inputdf2 = train_out_of_sample[['vol_now', 'ret_now']]
        predicted_y_train_out_of_sample = fittedModel.predict(np.array(inputdf2).reshape(len(inputdf2),2))
    elif forecaster == 3:
        fittedModel = Model.fit(np.array(train_in_sample.E_p).reshape(len(train_in_sample.E_p), 1),
                                np.array(train_in_sample.label))
        predicted_y_train_out_of_sample = fittedModel.predict(np.array(train_out_of_sample.E_p).reshape(len(train_out_of_sample.E_p), 1))
    elif forecaster == 4:
        inputdf = train_in_sample[['E_p', 'E_q']]
        fittedModel = Model.fit(np.array(inputdf).reshape(len(inputdf), 2),
                                np.array(train_in_sample.label))
        inputdf2 = train_out_of_sample[['E_p', 'E_q']]
        predicted_y_train_out_of_sample = fittedModel.predict(np.array(inputdf2).reshape(len(inputdf2), 2))
    elif forecaster == 5:
        inputdf = train_in_sample[['vol_now','ret_now', 'vol_sqr_now']]
        fittedModel = Model.fit(np.array(inputdf).reshape(len(inputdf),2),
                                np.array(train_in_sample.label))
        inputdf2 = train_out_of_sample[['vol_now','ret_now', 'vol_sqr_now']]
        predicted_y_train_out_of_sample = fittedModel.predict(np.array(inputdf2).reshape(len(inputdf2),2))
    elif forecaster == 6:
        inputdf = train_in_sample[['vol_now',  'ret_now','volxret_now','vol_sqr_now']]
        fittedModel = Model.fit(np.array(inputdf).reshape(len(inputdf),2),
                                np.array(train_in_sample.label))
        inputdf2 = train_out_of_sample[['vol_now',  'ret_now','volxret_now','vol_sqr_now']]
        predicted_y_train_out_of_sample = fittedModel.predict(np.array(inputdf2).reshape(len(inputdf2),2))

    element1 = np.array(train_out_of_sample.vol_now)
    element2 = 1 + benchmark_delta_star * predicted_y_train_out_of_sample
    predicted_train_out_of_sample_vol_future = pd.Series([a * b for a, b in zip(element1, element2)])
    observed_train_out_of_sample_vol_future = pd.Series(train_out_of_sample.vol_future)

    # measuring prediction performance in the training our of sample
    Performance_ = PerformanceMeasure()
    MSE = Performance_.mean_se(observed_train_out_of_sample_vol_future,predicted_train_out_of_sample_vol_future)
    return MSE
