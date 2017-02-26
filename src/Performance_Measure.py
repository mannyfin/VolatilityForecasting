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



        # import numpy as np
        #
        # # Squared error
        # SE = (y_fit.reshape(len(y), 1) - y.reshape(len(y), 1)) ** 2
        # # plt.figure(n)
        # plt.figure(se_plot.counter)
        # plt.plot(SE)

        #
        # plt.xlabel("t")
        # plt.ylabel("SE")
        #
        # # TODO change x-axis to time series
        #
        # '''
        # using the formula QL
        # '''
        # #
        # # value = y_fit1.reshape(len(y), 1) / y.reshape(len(y), 1)
        # # Ones = np.ones(len(y))
        # #
        # # (1 / len(y)) * (np.sum(value - np.log(value) - Ones.reshape(len(y), 1)))
        #
        # # # this only works with single parameter LR
        # # plt.scatter(x, y, color='black')
        # # plt.plot(x, y_fit1, color='blue', linewidth=3)
        # # plt.xticks(())
        # # plt.yticks(())
        # print("hi")