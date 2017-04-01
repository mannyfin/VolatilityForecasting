import numpy as np
from sklearn.linear_model import LogisticRegression as Logit
from Performance_Measure import *
from SEplot import se_plot as SE
import pandas as pd
import matplotlib.pyplot as plt

# adding labels and obtaining the training and the test samples for a given Delta value
def Obtain_Traing_Test(df, Delta):
    """
    :param df: columns including Date, V(seperating training and test samples), ret_past, vol_past, vol_now, vol_future
    :param Delta: Delta value which is a candidate of the optimized Delta
    :return: the training and test sample
    """
    df['label'] = 0

    # labeling
    values1 = abs(df.vol_now - df.vol_past * (1 + Delta))
    values2 = abs(df.vol_now - df.vol_past * (1 - Delta))
    condition = values1 < values2
    df.loc[condition, 'label'] = 1
    df.loc[~condition, 'label'] = -1

    # seperate data into training and test samples
    condition2 = df.V == 1
    df_training = df.loc[condition2]
    df_training = df_training.reset_index()
    df_test = df.loc[~condition2]
    df_test = df_test.reset_index()
    return df_training, df_test

# volatility prediction for training/test sample
def PredictVol(preprocess, Delta, warmup, train_or_test):
    """
    :param preprocess: the data frame created in main.py by returnvoldf.py
    :param Delta: Delta value which is a candidate of the optimized Delta
    :param warmup: the number of observations as a warm-up period for the model
    :param train_or_test: a string of "train" or "test"
    :return: all predicted volatilities
    """
    if train_or_test == "train":
        df_whole = Obtain_Traing_Test(preprocess, Delta)[0]
    elif train_or_test == "test":
        df_whole = Obtain_Traing_Test(preprocess, Delta)[1]

    PredictedVols = []
    for i in range(np.shape(df_whole)[0]-warmup+2):
        # model fitting and making predictions
        df = df_whole[:warmup-2+i]
        Model = Logit()
        Model.fit(np.array(df.vol_now).reshape(len(df.vol_now),1), np.array(df.label))
        predicted_y_t = Model.predict(df.vol_future.iloc[-1])
        vol_future_pred = df.vol_future.iloc[-1] *(1+predicted_y_t*Delta)
        PredictedVols.append(vol_future_pred)
    vol_future_observed = df.vol_future[warmup-2:]
    return PredictedVols, vol_future_observed

# Calculating MSE and QL for training/test sample
def Logit_MSE_QL(preprocess, Delta,warmup, train_or_test):
    """
    :param preprocess: the data frame created in main.py by returnvoldf.py
    :param Delta: Delta value which is a candidate of the optimized Delta
    :param warmup: the number of observations as a warm-up period for the model
    :param train_or_test: a string of "train" or "test"
    :return: the MSE and QL to measure the prediction performance
    """
    # model fitting and making predictions
    Performance_ = PerformanceMeasure()
    PredictedVols = PredictVol(preprocess, Delta, warmup, train_or_test)

    MSE = Performance_.mean_se(observed=preprocess.vol_future, prediction=PredictedVols)
    QL = Performance_.quasi_likelihood(observed=preprocess.vol_future, prediction=PredictedVols)
    return MSE, QL

# optimize in the training sample
def Optimize(df, DeltaSeq,filename):
    """
    :param df: the data frame of training sample
    :param DeltaSeq: a sequence of Delta values
    :return: the optimized Delta
    """
    MSEs = []
    for i in range(len(DeltaSeq)):
        MSEs.append(Logit_MSE_QL(df, DeltaSeq[i])[0])

    minIndex = MSEs.index(min(MSEs))
    OptimalDelta = DeltaSeq[minIndex]

    plt.plot(DeltaSeq,MSEs)
    plt.xlabel('Delta')
    plt.ylabel('MSE')
    plt.title(str(filename)+' Delta against MSE')
    plt.show()
    return OptimalDelta




DeltaSeq = np.linspace(0.1, 1, num=10)
optimalDelta = Optimize(df=preprocess, DeltaSeq,filename="AUDUSD.csv")
PredictVol(df, optimalDelta)

