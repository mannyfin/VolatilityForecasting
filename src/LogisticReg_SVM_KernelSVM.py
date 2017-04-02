import numpy as np
from sklearn.linear_model import LogisticRegression as Logit
from Performance_Measure import *
from SEplot import se_plot as SE
import pandas as pd
import matplotlib.pyplot as plt
import forecaster_classifier as fc
from sklearn.svm import SVC as SVC

# adding labels and obtaining the training and the test samples for a given Delta value
def Obtain_Traing_Test(df, Delta, forecaster, p=None,q=None):
    """
    :param df: columns including Date, V(seperating training and test samples), ret_past, vol_past, vol_now, vol_future
    :param Delta: Delta value which is a candidate of the optimized Delta
    :param forecaster: forecaster = 1,2,3 or 4
    :param p: p is a parameter in forecaster 3 and forecaster 4 
    :param q: q is a parameter in forecaster 4
    :return: the training and test sample
    """
    # labeling
    # values1 = abs(df.vol_now - df.vol_past * (1 + Delta))
    # values2 = abs(df.vol_now - df.vol_past * (1 - Delta))
    # condition = values1 < values2
    # df.loc[condition, 'label'] = 1
    # df.loc[~condition, 'label'] = -1

    if forecaster==1:
        params = {'delta': Delta, 'vol_name': 'vol_past'}
        df = fc.forecaster_classifier(df,fxn=fc.volonly,params=params)

    elif forecaster==2:
        df = fc.forecaster_classifier(df, fxn=fc.volandret, params={'delta': Delta,
                                                                'vol_name': 'vol_past',
                                                                'ret_name': 'ret_past'})
    elif forecaster==3:
        df = fc.forecaster_classifier(df, fxn=fc.volonly, params={'delta': Delta,
                                                                    'vol_name': 'vol_past',
                                                                    'p': p})

    elif forecaster==4:
        df = fc.forecaster_classifier(df, fxn=fc.volandret, params={'delta': Delta,
                                                                    'vol_name': 'vol_past',
                                                                    'p': p,
                                                                    'q':q, 'ret_name':'ret_past'})

    # seperate data into training and test samples
    condition2 = df.V == 1
    df_training = df.loc[condition2]
    df_training = df_training.reset_index()
    df_test = df.loc[~condition2]
    df_test = df_test.reset_index()
    return df_training, df_test

# volatility prediction for training/test sample
def PredictVol(preprocess, Delta, warmup, train_or_test, model, deg=None, forecaster=None, p=None,q=None):
    """
    :param preprocess: the data frame created in main.py by returnvoldf.py
    :param Delta: Delta value which is a candidate of the optimized Delta
    :param warmup: the number of observations as a warm-up period for the model, which is 400 in our case
    :param train_or_test: a string of "train" or "test"
    :param model: model can take inputs "LogisticRegression", "SVM", "KernelSVM_poly" ,"KernelSVM_rbf" or "KernelSVM_sigmoid"
    :param deg: degree of the Kernel SVM when kernel="poly"
    :param forecaster: forecaster = 1,2,3 or 4
    :param p: p is a parameter in forecaster 3 and forecaster 4 
    :param q: q is a parameter in forecaster 4
    :return: all predicted volatilities
    """
    if train_or_test == "train":
        df_whole = Obtain_Traing_Test(preprocess, Delta, forecaster, p,q)[0]
    elif train_or_test == "test":
        df_whole = Obtain_Traing_Test(preprocess, Delta, forecaster, p,q)[1]

    # specify the type of the model
    if model =="LogisticRegression":
        Model = Logit()
    elif model == "SVM":
        Model = SVC(kernel='linear')
    elif model == "KernelSVM_poly":
        Model = SVC(kernel='poly', degree=deg)
    elif model == "KernelSVM_rbf":
        Model = SVC(kernel='rbf')
    elif model == "KernelSVM_sigmoid":
        Model = SVC(kernel='sigmoid')


    PredictedVols = []
    for i in range(np.shape(df_whole)[0]-warmup+2):
        # model fitting and making predictions
        df = df_whole[:warmup-2+i]
        Model.fit(np.array(df.vol_now).reshape(len(df.vol_now),1), np.array(df.label))
        predicted_y_t = Model.predict(df.vol_future.iloc[-1])
        vol_future_pred = df.vol_future.iloc[-1] *(1+predicted_y_t*Delta)
        PredictedVols.append(vol_future_pred[0])

    vol_future_observed = df_whole.vol_future[warmup-2:]
    return pd.Series(PredictedVols), pd.Series(vol_future_observed)

# Calculating MSE and QL for training/test sample
def MSE_QL(preprocess, Delta,warmup, train_or_test, model, deg=None, forecaster=None, p=None,q=None):
    """
    :param preprocess: the data frame created in main.py by returnvoldf.py
    :param Delta: Delta value which is a candidate of the optimized Delta
    :param warmup: the number of observations as a warm-up period for the model, which is 400 in our case
    :param train_or_test: a string of "train" or "test"
    :param model: model can take inputs "LogisticRegression", "SVM", "KernelSVM_poly" ,"KernelSVM_rbf" or "KernelSVM_sigmoid"
    :param deg: degree of the Kernel SVM when kernel="poly"
    :param forecaster: forecaster = 1,2,3 or 4
    :param p: p is a parameter in forecaster 3 and forecaster 4 
    :param q: q is a parameter in forecaster 4
    :return: the MSE and QL to measure the prediction performance
    """
    # model fitting and making predictions
    Performance_ = PerformanceMeasure()
    PredictedVols = PredictVol(preprocess, Delta, warmup, train_or_test, model, deg, forecaster, p,q)
    prediction = PredictedVols[0]
    observed  = PredictedVols[1]

    MSE = Performance_.mean_se(observed, prediction)
    QL = Performance_.quasi_likelihood(observed.astype('float64').astype('float64'), prediction.astype('float64'))
    return MSE, QL,prediction,observed

# optimize in the training sample
def Optimize(preprocess, DeltaSeq,warmup, filename, model, deg=None, forecaster=None, p=None,q=None):
    """
    :param preprocess: the data frame created in main.py by returnvoldf.py
    :param DeltaSeq: a sequence of Delta values
    :param warmup: the number of observations as a warm-up period for the model, which is 400 in our case
    :param model: model can take inputs "LogisticRegression", "SVM", "KernelSVM_poly" ,"KernelSVM_rbf" or "KernelSVM_sigmoid"
    :param deg: degree of the Kernel SVM when kernel="poly"
    :param forecaster: forecaster = 1,2,3 or 4
    :param p: p is a parameter in forecaster 3 and forecaster 4 
    :param q: q is a parameter in forecaster 4
    :return: the optimized Delta
    """
    MSEs = []
    train_or_test = "train"
    for i in range(len(DeltaSeq)):
        MSEOutput = MSE_QL(preprocess, DeltaSeq[i], warmup, train_or_test, model, deg, forecaster, p,q)[0]
        MSEs.append(MSEOutput)
    # find the index of the minimum MSE
    minIndex = MSEs.index(min(MSEs))
    OptimalDelta = DeltaSeq[minIndex]
    # Find the index of min p and q based on line 142

    """
    # plot of MSE vs log Delta
    # make different plots depending on different forecaster method. i.e. for forecaster=1, plot mse vs log delta
    # for forecaster = 3, plot mse vs delta vs p
    # for forecaster = 4, plot mse vs delta vs p vs q
    """
    plt.plot(np.log(DeltaSeq),MSEs)
    plt.xlabel('log(Delta)')
    plt.ylabel('MSE')
    plt.title(str(filename)+' MSE against log(Delta)')
    plt.show()
    return OptimalDelta


# measure the prediction performance in the test sample
def MSE_QL_SE_Test(preprocess,DeltaSeq,warmup_test, filename, model, deg=None, forecaster=None, p=None,q=None):
    """
    :param preprocess: the data frame created in main.py by returnvoldf.py
    :param DeltaSeq: a sequence of Delta values
    :param warmup: the number of observations as a warm-up period for the model, which is 400 in our case
    :param model: model can take inputs "LogisticRegression", "SVM", "KernelSVM_poly" ,"KernelSVM_rbf" or "KernelSVM_sigmoid"
    :param deg: degree of the Kernel SVM when kernel="poly"
    :param forecaster: forecaster = 1,2,3 or 4
    :param p: p is a parameter in forecaster 3 and forecaster 4 
    :param q: q is a parameter in forecaster 4
    :return: 
    """
    warmup_train = 400
    OptimalDelta = Optimize(preprocess, DeltaSeq,warmup_train, filename, model, deg, forecaster=forecaster)

    train_or_test = "test"
    Output = MSE_QL(preprocess, OptimalDelta,warmup_test, train_or_test, model, deg,forecaster, p,q)
    MSE_test = Output[0]
    QL_test = Output[1]
    prediction  = Output[2]
    observed = Output[3]

    df_test = Obtain_Traing_Test(preprocess, OptimalDelta, forecaster, p,q)[1]
    """ return a plot of the squared error"""
    SE(observed, prediction, df_test.Date[warmup-2:])
    plt.title(str(filename) + '_Squared Error_Logistic Regression')
    plt.show()

    return MSE_test, QL_test

    # DeltaSeq = np.exp(np.linspace(-10, -2, num=100))
    #
    # TestResult_Logit = MSE_QL_SE_Test(preprocess, DeltaSeq, warmup_test=100, filename="AUDUSD", model="LogisticRegression",forecaster=1)
    # TestResult_SVM = MSE_QL_SE_Test(preprocess, DeltaSeq,warmup=100, filename="AUDUSD", model="SVM", forecaster=4, p=3, q=2)
    # TestResult_KernelSVM_poly = MSE_QL_SE_Test(preprocess, DeltaSeq,warmup_test=100, filename="AUDUSD", model="KernelSVM_poly", deg=3,
    #                                            forecaster=4, p=3, q=2)
    # TestResult_KernelSVM_rbf = MSE_QL_SE_Test(preprocess, DeltaSeq,warmup_test=100, filename="AUDUSD", model="KernelSVM_rbf",
    #                                           forecaster=4, p=3, q=2)
    # TestResult_KernelSVM_sigmoid = MSE_QL_SE_Test(preprocess, DeltaSeq,warmup_test=100, filename="AUDUSD", model="KernelSVM_sigmoid",
    #                                               forecaster=4, p=3, q=2)
