from sklearn.metrics import mean_squared_error as mse


class PerformanceMeasure(object):

    def MeanSE(self, observed,prediction):
        PerformanceMeasure.MSE = mse(observed, prediction)
        return PerformanceMeasure.MSE

    def Quasi_Liklihood(self, observed, prediction):

        # TODO QL DOES NOT WORK DUE TO ZEROES IN DATA SERIES
        #
        # value = y_fit1.reshape(len(y), 1) / y.reshape(len(y), 1)
        # Ones = np.ones(len(y))
        #
        # PerformanceMeasure.QL = (1 / len(y)) * (np.sum(value - np.log(value) - Ones.reshape(len(y), 1)))

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
        # # TODO QL DOES NOT WORK DUE TO ZEROES IN DATA SERIES
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
        print("hi")