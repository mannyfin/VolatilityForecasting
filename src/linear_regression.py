import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression as sklr
from sklearn.metrics import mean_squared_error as mse
from SEplot import se_plot as SE
from sklearn.linear_model import LinearRegression as lr

class LinRegression:
    from sklearn.linear_model import LinearRegression as sklr
    # def __init__(self, data):
    #     """
    #     :param daily_vol_result: result from def daily_vol_calc
    #     """
    #     import numpy as np
    #     # self.data = np.asarray(data['Volatility_Daily'])
    #     self.data = data
    #     # self.data = data
    # TODO FIX LR.FIT POSITIONAL ARG FOR Y

# @staticmethod
    def one_day_trailing(data):
        """
        Compute one day trailing volatility
        :return: Mean Squared error, slope: b, and y-int: c
        """

        fitting = sklr
        data = np.asarray(data['Volatility_Daily'])
        x = data[:-1]
        y = data[1:]
        x = x.reshape(len(x), 1)
        A = lr()
        A.fit(x, y)
        b = A.coef_[0]
        c = A.intercept_
        y_fit1 = b * x + c
        MSE1 = mse(y, y_fit1)
        print("MSE1 is " + str(MSE1))
        print("intercept is " + str(c))
        print("slope is " + str(b))

        SE(y, y_fit1, 1)

        return MSE1, b, c

    def lin_reg(data, n):
        data = np.asarray(data['Volatility_Daily'])
        # x = [data[(n-i-1):(-i-1)].reshape(len(data[(n-i-1):(-i-1)]), 1) for i in range(n)]
        x = [i for i in range(n)]
        for i in range(n):
            x[i] = data[(n - i - 1):(-i - 1)]
            x[i] = x[i].reshape(len(x[i]), 1)
        x = np.column_stack(x)
        # x[i] = data[(n-i-1):(-i-1)]
        y = data[n:]
        A = lr()
        A.fit(x, y)
        b = [A.coef_[i] for i in range(n)]
        c = A.intercept_
        yfit = np.matmul(x, b) + c

        MSE = mse(y, yfit)
        SE(y, yfit, n)
        print(str(n)+" Lag's "+"MSE is " + str(MSE))

        return MSE, b, c

    # def three_day_trailing(data):
    #     """
    #     use past 3 volatilties to predict tomorrow’s volatiltiy
    #     y represents tomorrow's volatility;
    #     x1, x2 and x3 represent past 3 volatilties
    #
    #     """
    #     data = np.asarray(data['Volatility_Daily'])
    #     y = data[n:]
    #     x3, x2, x1, y = data[2:-1], data[1:-2], data[:-3], data[3:]
    #     x1 = x1.reshape(len(x1), 1)
    #     x2 = x2.reshape(len(x2), 1)
    #     x3 = x3.reshape(len(x3), 1)
    #     x = np.c_[x1, x2, x3]
    #     A = lr()
    #     A.fit(x, y)
    #     b1 = A.coef_[0]
    #     b2 = A.coef_[1]
    #     b3 = A.coef_[2]
    #
    #     c3 = A.intercept_
    #
    #     # there's a more general way to do this but w/e for now
    #     y_fit3 = b1*x1 + b2*x2 + b3*x3 + c3
    #     # Calculate the MSE
    #     MSE3 = mse(y, y_fit3)
    #     SE(y, y_fit3, 3)
    #
    #     print("MSE3 is " + str(MSE3))
    #     print("intercept3 is " + str(c3))
    #     print("slope3 is " + str(b3))
    #
    #     return c3, b1, b2, b3, MSE3
    #
    # def five_day_trailing(data):
    #     """
    #     use past 3 volatilties to predict tomorrow’s volatiltiy
    #     y represents tomorrow's volatility;
    #     x1, x2 and x3 represent past 3 volatilties
    #
    #     """
    #     data = np.asarray(data['Volatility_Daily'])
    #     x1, x2, x3, x4, x5, y = data[4:-1], data[3:-2], data[2:-3], data[1:-4], \
    #                             data[:-5], data[5:]
    #     x1 = x1.reshape(len(x1), 1)
    #     x2 = x2.reshape(len(x2), 1)
    #     x3 = x3.reshape(len(x3), 1)
    #     x4 = x4.reshape(len(x4), 1)
    #     x5 = x5.reshape(len(x5), 1)
    #     x = np.c_[x1, x2, x3, x4, x5]
    #     A = lr()
    #     A.fit(x, y)
    #     b1 = A.coef_[0]
    #     b2 = A.coef_[1]
    #     b3 = A.coef_[2]
    #     b4 = A.coef_[3]
    #     b5 = A.coef_[4]
    #
    #     c5 = A.intercept_
    #
    #     # there's a more general way to do this but w/e for now
    #     y_fit5 = b1 * x1 + b2 * x2 + b3 * x3 + b4 * x4 + b5 * x5 + c5
    #     # Calculate the MSE
    #     MSE5 = mse(y, y_fit5)
    #     SE(y, y_fit5, 5)
    #     print("MSE5 is " + str(MSE5))
    #     print("intercept5 is " + str(c5))
    #     print("slope5 for b5 is " + str(b5))
    #     return c5, b1, b2, b3, b4, b5, MSE5
