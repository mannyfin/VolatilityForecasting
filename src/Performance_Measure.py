from sklearn.metrics import mean_squared_error as mse
import numpy as np
import pandas as pd

class PerformanceMeasure(object):
    QL = 0
    MSE = 0

    def __init__(self, QL = 0, MSE= 0):
        self.QL = QL
        self.MSE = MSE

    def mean_se(self, observed, prediction):



        if isinstance(observed, pd.core.frame.DataFrame) & isinstance(prediction, pd.core.frame.DataFrame):
            self.MSE = np.mean(np.square(np.subtract(observed, prediction)))
            mse_sum = pd.Series(sum(self.MSE), index=['SumMSE'])
            self.MSE = self.MSE.append(mse_sum)

            # this line converts to df and transposes from cols to rows
            self.MSE = pd.DataFrame(self.MSE).T
            print("MSE is: \n" + str(self.MSE))
        else:
            self.MSE = mse(observed, prediction)
            self.MSE = self.MSE
            print("MSE is: " + str(self.MSE))

        return self.MSE

    def quasi_likelihood(self, observed, prediction):
        """Note: QL DOES NOT WORK IF THERE ARE ZEROES IN DATA SERIES"""

        # if len(prediction.index[prediction<0]) >0:
        #     observed = observed.drop(observed.first_valid_index() + prediction.index[prediction < 0])
        #     prediction = prediction[prediction>0]

        if isinstance(observed, pd.core.series.Series) & isinstance(prediction, pd.core.series.Series):

            # the line below will only work if the data is not a dataframe
            value = observed.ravel() / prediction.ravel()
            ones = np.ones(len(observed))
            self.QL = (1 / len(observed)) * (np.sum(value - np.log(value) - ones))
            print("QL is: " + str(self.QL))
        elif isinstance(observed,pd.core.frame.DataFrame) & isinstance(prediction,pd.core.frame.DataFrame):
            # the line below will only work if the data is not a dataframe
            value = np.divide(observed,prediction)
            ones = np.ones(np.shape(observed))
            self.QL = (1 / len(observed)) * (np.sum(value - np.log(value) - ones))
            ql_sum = pd.Series(sum(self.QL), index=['SumQL'])
            self.QL = self.QL.append(ql_sum)

            # this line converts to df and transposes from cols to rows
            self.QL = pd.DataFrame(self.QL).T

            print("QL is: \n" + str(self.QL))

        # print("QL is: " + str(self.QL))
        return self.QL

    # def quasi_likelihood(self, observed, prediction):
    #     """Note: QL DOES NOT WORK IF THERE ARE ZEROES IN DATA SERIES"""
    #
    #     if len(prediction.index[prediction<0]) >0:
    #         observed = observed.drop(observed.first_valid_index() + prediction.index[prediction < 0])
    #         prediction = prediction[prediction>0]
    #
    #     value = prediction.ravel() / observed.ravel()
    #     ones = np.ones(len(observed))
    #
    #
    #     self.QL = (1 / len(observed)) * (np.sum(value - np.log(value) - ones))
    #     print("QL is: " + str(self.QL))
    #     return self.QL
