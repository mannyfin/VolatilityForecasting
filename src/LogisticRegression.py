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
    :return: the predicted volatilities
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
    df_training = df_training.drop('index', 1)
    df_test = df.loc[~condition2]
    df_test = df_test.reset_index()
    df_test = df_test.drop('index', 1)
    return df_training, df_test


# volatility prediction for training/test sample
def PredictVol(df, Delta):
    """
    :param df: can be training or test sample
    :param Delta: Delta value which is a candidate of the optimized Delta
    :return: the predicted volatilities
    """
    # model fitting and making predictions
    Model = Logit()
    Model.fit(np.array(df.vol_past).reshape(len(df.vol_past),1), np.array(df.label))
    predicted_y_t = Model.predict(np.array(df.vol_now).reshape(len(df.vol_now), 1))
    df['predicted_y_t'] = pd.Series(predicted_y_t, index=df.index)
    df['vol_future_pred'] = df.vol_now * (1 + Delta * df.predicted_y_t)
    return df.vol_future_pred

# Calculating MSE and QL for training/test sample
def Logit_MSE_QL(df, Delta):
    """
    :param df: can be training or test sample
    :param Delta: Delta value which is a candidate of the optimized Delta
    :return: the MSE and QL to measure the prediction performance
    """
    # model fitting and making predictions
    Performance_ = PerformanceMeasure()
    PredictedVol = PredictVol(df, Delta)
    MSE = Performance_.mean_se(observed=df.vol_future, prediction=PredictedVol)
    QL = Performance_.quasi_likelihood(observed=df.vol_future, prediction=PredictedVol)
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

    plt.plot(DeltaSeq,MSEs)
    plt.xlabel('Delta')
    plt.ylabel('MSE')
    plt.title(str(filename)+' Delta against MSE')
    plt.show()


DeltaSeq = np.linspace(0.1, 1, num=10)


