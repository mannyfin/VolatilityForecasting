from sklearn.metrics import mean_squared_error as mse
import numpy as np


class PerformanceMeasure(object):
    QL = 0
    MSE = 0

    def __init__(self, QL = 0, MSE= 0):
        self.QL = QL
        self.MSE = MSE

    def mean_se(self, observed, prediction):
        self.MSE = mse(observed, prediction)
        print("MSE is: " + str(self.MSE))
        return self.MSE

    def quasi_likelihood(self, observed, prediction):
        """Note: QL DOES NOT WORK IF THERE ARE ZEROES IN DATA SERIES"""

        value = prediction.reshape(len(observed), 1) / observed.reshape(len(observed), 1)
        ones = np.ones(len(observed))

        self.QL = (1 / len(observed)) * (np.sum(value - np.log(value) - ones.reshape(len(observed), 1)))
        print("QL is: " + str(self.QL))
        return self.QL
