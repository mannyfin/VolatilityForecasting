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
    dfabc = df.copy()
    if forecaster==1:
        params = {'delta': Delta, 'vol_name': 'vol_past'}
        dfabc = fc.forecaster_classifier(dfabc,fxn=fc.volonly,params=params)

    elif forecaster==2:
        dfabc = fc.forecaster_classifier(dfabc, fxn=fc.volandret, params={'delta': Delta,
                                                                'vol_name': 'vol_past',
                                                                'ret_name': 'ret_past'})
    elif forecaster==3:
        dfabc = fc.forecaster_classifier(dfabc, fxn=fc.volonly, params={'delta': Delta,
                                                                    'vol_name': 'vol_past',
                                                                    'p': p})

    elif forecaster==4:
        dfabc = fc.forecaster_classifier(dfabc, fxn=fc.volandret, params={'delta': Delta,
                                                                    'vol_name': 'vol_past',
                                                                    'p': p,
                                                                    'q':q, 'ret_name':'ret_past'})

    # seperate data into training and test samples
    condition2 = df.V == 1
    df_training = dfabc.loc[condition2]
    df_training = df_training.reset_index()
    df_test = dfabc.loc[~condition2]
    df_test = df_test.reset_index()
    del dfabc
    return df_training, df_test

# volatility prediction for training/test sample
def PredictVol(preprocess_predict, Delta, warmup, train_or_test, model, deg=None, forecaster=None, p=None,q=None):
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
        df_whole = Obtain_Traing_Test(preprocess_predict, Delta, forecaster, p,q)[0]
    elif train_or_test == "test":
        df_whole = Obtain_Traing_Test(preprocess_predict, Delta, forecaster, p,q)[1]

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
        df = df_whole[i:warmup-2+i]
        Model.fit(np.array(df.vol_now).reshape(len(df.vol_now),1), np.array(df.label))
        predicted_y_t = Model.predict(df.vol_future.iloc[-1])
        vol_future_pred = df.vol_future.iloc[-1] *(1+predicted_y_t*Delta)
        PredictedVols.append(vol_future_pred[0])

    vol_future_observed = df_whole.vol_future[warmup-2:]
    return pd.Series(PredictedVols), pd.Series(vol_future_observed)


# Calculating MSE and QL for training/test sample
def MSE_QL(preprocess_data_input, Delta,warmup, train_or_test, model, deg=None, forecaster=None, p=None,q=None):
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
    PredictedVols = PredictVol(preprocess_data_input, Delta, warmup, train_or_test, model, deg, forecaster, p,q)
    prediction = PredictedVols[0]
    observed  = PredictedVols[1]

    MSE = Performance_.mean_se(observed, prediction)
    QL = Performance_.quasi_likelihood(observed.astype('float64').astype('float64'), prediction.astype('float64'))
    return MSE, QL,prediction,observed


# optimize in the training sample
def Optimize(preprocess_data, DeltaSeq,warmup, filename, model, deg=None, forecaster=None, p_seq=None,q_seq=None, stringinput=None):
    """
    :param preprocess: the data frame created in main.py by returnvoldf.py
    :param DeltaSeq: a sequence of Delta values
    :param warmup: the number of observations as a warm-up period for the model, which is 400 in our case
    :param model: model can take inputs "LogisticRegression", "SVM", "KernelSVM_poly" ,"KernelSVM_rbf" or "KernelSVM_sigmoid"
    :param deg: degree of the Kernel SVM when kernel="poly"
    :param forecaster: forecaster = 1,2,3 or 4
    :param p_seq: p_seq contains the possible values of the parameter p in forecaster 3 and forecaster 4
    :param q_seq: q_seq contains the possible values of the parameter q in forecaster 4
    :return: the optimized Delta
    """
    MSEs = []
    Delta_values_seq = []
    if p_seq is None: p_seq = [None]
    if q_seq is None: q_seq = [None]
    p_values_seq = []
    q_values_seq = []
    train_or_test = "train"
    for i in range(len(DeltaSeq)):
        for j in range(len(p_seq)):
            for k in range(len(q_seq)):
                MSEOutput = MSE_QL(preprocess_data, DeltaSeq[i], warmup, train_or_test, model, deg, forecaster, p_seq[j],q_seq[k])[0]
                MSEs.append(MSEOutput)
                Delta_values_seq.append(DeltaSeq[i])
                p_values_seq.append(p_seq[j])
                q_values_seq.append(q_seq[k])
    # find the index of the minimum MSE
    minIndex = MSEs.index(min(MSEs))
    OptimalDelta = Delta_values_seq[minIndex]
    optimal_p = p_values_seq[minIndex]
    optimal_q = q_values_seq[minIndex]

    # TODO do the stuff in the comments below
    """
    # plot of MSE vs log Delta
    # make different plots depending on different forecaster method. i.e. for forecaster=1 or 2 plot mse vs log(delta)
    # for forecaster = 3, plot mse vs log(delta) vs p
    # for forecaster = 4, plot mse vs log(delta) vs p vs q
    """
    plt.plot(np.log(DeltaSeq),MSEs)
    plt.xlabel('log(Delta)')
    plt.ylabel('MSE')

    if forecaster == 1 or 2:
        title = str(filename) + ' ' + str(stringinput) + ' ' + str(model) + ' MSE against log(Delta)'

    if forecaster == 3:
        title = str(filename) + ' ' + str(stringinput) + ' ' + str(model) + ' MSE against log(Delta) p=' + str(p)
    if forecaster == 4:
        title = str(filename) + ' ' + str(stringinput) + ' ' + str(model) + ' MSE against log(Delta) p=' \
                + str(p) + ' q=' + str(q)

    plt.title(title)
    # save the figs
    plt.savefig(title+'.png')
    # plt.show()
    plt.close()
    return OptimalDelta,optimal_p,optimal_q


# measure the prediction performance in the test sample
def MSE_QL_SE_Test(preprocess_info,DeltaSeq,warmup_test, filename, model, deg=None, forecaster=None, p_seq=None,q_seq=None,stringinput=None):
    """
    :param preprocess: the data frame created in main.py by returnvoldf.py
    :param DeltaSeq: a sequence of Delta values
    :param warmup: the number of observations as a warm-up period for the model, which is 400 in our case
    :param model: model can take inputs "LogisticRegression", "SVM", "KernelSVM_poly" ,"KernelSVM_rbf" or "KernelSVM_sigmoid"
    :param deg: degree of the Kernel SVM when kernel="poly"
    :param forecaster: forecaster = 1,2,3 or 4
    :param p_seq: p_seq contains the possible values of the parameter p in forecaster 3 and forecaster 4
    :param q_seq: q_seq contains the possible values of the parameter q in forecaster 4
    :return: 
    """
    if stringinput == 'Daily':
        warmup_train = 100 # for daily
    elif stringinput == 'Weekly':
        warmup_train = 50 # for weekly

    # train the model
    OptimizationOutput = Optimize(preprocess_info, DeltaSeq,warmup_train, filename, model, deg, p_seq=p_seq, q_seq=q_seq, forecaster=forecaster,
                            stringinput=stringinput)
    OptimalDelta = OptimizationOutput[0]
    Optimal_p = OptimizationOutput[1]
    Optimal_q = OptimizationOutput[2]
    # test the model
    train_or_test = "test"
    Output = MSE_QL(preprocess_info, OptimalDelta, warmup_test, train_or_test, model, deg,forecaster, Optimal_p,Optimal_q)
    MSE_test = Output[0]
    QL_test = Output[1]
    print('MSE_test is ' + str(MSE_test))
    print('QL_test is '+ str(QL_test))
    prediction = Output[2]
    observed = Output[3]

    df_test = Obtain_Traing_Test(preprocess_info, OptimalDelta, forecaster, Optimal_p,Optimal_q)[1]
    """ return a plot of the squared error"""
    df_test["Date"] = df_test.index
    SE(observed, prediction, df_test.Date[warmup_test-2:])
    if forecaster == 1 or 2:
        title = str(filename) + ' ' + str(stringinput) + ' ' + str(model)+'_Squared Error_Logistic Regression'
    # save the figs
    elif forecaster == 3:
        title = str(filename) + ' ' + str(stringinput) + ' ' + str(model) + '_Squared Error_Logistic Regression p=' + str(p)
    elif forecaster == 4:
        title = str(filename) + ' ' + str(stringinput) + ' ' + str(model) + '_Squared Error_Logistic Regression p=' + \
                str(p) + ' q='+str(q)

    plt.title(title)
    plt.savefig(title+'.png')
    # plt.show()
    plt.close()
    return MSE_test, QL_test
