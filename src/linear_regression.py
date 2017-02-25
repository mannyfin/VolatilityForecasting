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
        SE(y, yfit)
        plt.title(str(n) + " Day Lag's SE: Linear Regression ")
        # print(str(n) + " Lag's " + "MSE is " + str(MSE))
        # plt.show()

        return MSE, b, c
# @staticmethod
#     def one_day_trailing(data):
#         """
#         Compute one day trailing volatility
#         :return: Mean Squared error, slope: b, and y-int: c
#         """
#
#         fitting = sklr
#         data = np.asarray(data['Volatility_Daily'])
#         x = data[:-1]
#         y = data[1:]
#         x = x.reshape(len(x), 1)
#         A = lr()
#         A.fit(x, y)
#         b = A.coef_[0]
#         c = A.intercept_
#         y_fit1 = b * x + c
#         MSE1 = mse(y, y_fit1)
#         print("MSE1 is " + str(MSE1))
#         print("intercept is " + str(c))
#         print("slope is " + str(b))
#
#         SE(y, y_fit1, 1)
#
#         return MSE1, b, c



