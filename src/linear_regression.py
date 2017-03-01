import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from sklearn.metrics import mean_squared_error as mse

import matplotlib.pyplot as plt
import numpy as np

# from sklearn.metrics import mean_squared_error as mse
from SEplot import se_plot as SE
from sklearn.linear_model import LinearRegression as lr
from Performance_Measure import *


class LinRegression:

    def lin_reg(data, n, filename, stringinput, warmup_period):
        """
        :param warmup_period uses a fixed window warmup period defined by the var, warmup_period
        :param data
        :param n is the number of regressors
        :param filename is the filename, .csv
        :param stringinput is the string that goes on the plot
        """
        # ysave=[]


        # for warmup_period in
        # # training set, x
        # for i in range(n):
        #     x[i] = data['Volatility_Time'][i:(warmup_period + i + 1)]
        #     # x[i] = x[i].reshape(len(x[i]), 1)data['Volatility_Time'][i:(warmup_period + i + 1)
        # x = np.column_stack(x)
        # # x[i] = data[(n-i-1):(-i-1)]
        # y = data['Volatility_Time'][n:n+warmup_period+1]
        # A = lr()
        # A.fit(x, y)
        # b = [A.coef_[i] for i in range(n)]
        # c = A.intercept_
        # # yfit = np.matmul(x, b) + c
        # # reshape data for prediction
        # A.predict(data['Volatility_Time'][13:16].values.reshape(1, -1))
        # # #
        # x = [i for i in range(n)]
        data = data.reset_index(range(len(data)))
        prediction=[]
        x = [i for i in range(n)]
        for initial in range(warmup_period, len(data['Volatility_Time'])-n):
            for i in range(n):
                x[i] = data['Volatility_Time'][i:(initial +i)]
                # x[i] = x[i].reshape(len(x[i]), 1)
            xstacked = np.column_stack(x)
            # x[i] = data[(n-i-1):(-i-1)]
            y = data['Volatility_Time'][n:n+initial]
            A = lr()
            A.fit(xstacked, y)
            b = [A.coef_[i] for i in range(n)]
            c = A.intercept_
            # yfit = np.matmul(x, b) + c
            # # reshape data for prediction
            prediction.append(A.predict(data.Volatility_Time[initial:n+initial].values.reshape(1, -1))[0])
            # prediction.append(A.predict(y[y.last_valid_index()])[0])

        #     if n == 1:
        #         mat1 = np.transpose([c, b[0]])
        #         mat2 = [1, y[y.last_valid_index()]]
        #         prediction.append(np.matmul(mat1, mat2))
        #     if n == 3:
        #         mat1 = np.transpose([c, b[0], b[1], b[2]])
        #         mat2 = [1, x[1][x[1].last_valid_index()], x[2][x[2].last_valid_index()], y[y.last_valid_index()]]
        #         prediction.append(np.matmul(mat1, mat2))
        #     if n == 5:
        #         mat1 = np.transpose([c, b[0], b[1], b[2], b[3], b[4]])
        #         mat2 = [1, x[1][x[1].last_valid_index()], x[2][x[2].last_valid_index()], x[3][x[3].last_valid_index()],
        #                 x[4][x[4].last_valid_index()], y[y.last_valid_index()]]
        #         prediction.append(np.matmul(mat1, mat2))
        #     if n == 10:
        #         mat1 = np.transpose([c, b[0], b[1], b[2], b[3], b[4], b[5],b[6],b[7],b[8],b[9]])
        #         mat2 = [1, x[1][x[1].last_valid_index()], x[2][x[2].last_valid_index()], x[3][x[3].last_valid_index()],
        #                 x[4][x[4].last_valid_index()], x[5][x[5].last_valid_index()],x[6][x[6].last_valid_index()],
        #                 x[7][x[7].last_valid_index()],x[8][x[8].last_valid_index()],x[9][x[9].last_valid_index()],
        #                 y[y.last_valid_index()]]
        #         prediction.append(np.matmul(mat1, mat2))
        #
        # y = y[(y.first_valid_index()+warmup_period-n-1):]
        y = data.Volatility_Time[warmup_period+n:]
        prediction = pd.Series(prediction)
        Performance_ = PerformanceMeasure()
        MSE = Performance_.mean_se(observed=y, prediction=prediction)
        # QL = Performance_.quasi_likelihood(observed=y, prediction=prediction)
        QL = Performance_.quasi_likelihood(observed=y, prediction=prediction)

        dates = data['Date'][n:]
        label=str(filename)+" "+str(stringinput)+" Linear Regression: "+str(n) + " Past Vol SE "
        SE(y, prediction, dates,function_method=label)

        # plt.title(str(filename)+" "+str(stringinput)+" Linear Regression: "+str(n) + " Past Vol SE ")

        return MSE, QL, b, c

# from SEplot import se_plot as SE
# from sklearn.linear_model import LinearRegression as lr
# from Performance_Measure import *
#
#
# class LinRegression:
#
#     def lin_reg(data, n, filename, stringinput, warmup_period):
#         """
#         :param warmup_period uses a fixed window warmup period defined by the var, warmup_period
#         :param data
#         :param n is the number of regressors
#         :param filename is the filename, .csv
#         :param stringinput is the string that goes on the plot
#         """
#
#         # ysave=[]
#         x = [i for i in range(n)]
#         # data['Volatility_Time'] = data['Volatility_Time'].reset_index(drop=True, inplace=True)
#         data = data.reset_index(range(len(data)))
#         # for warmup_period in
#         # # training set, x
#         # for i in range(n):
#         #     x[i] = data['Volatility_Time'][i:(warmup_period + i + 1)]
#         #     # x[i] = x[i].reshape(len(x[i]), 1)data['Volatility_Time'][i:(warmup_period + i + 1)
#         # x = np.column_stack(x)
#         # # x[i] = data[(n-i-1):(-i-1)]
#         # y = data['Volatility_Time'][n:n+warmup_period+1]
#         # A = lr()
#         # A.fit(x, y)
#         # b = [A.coef_[i] for i in range(n)]
#         # c = A.intercept_
#         # # yfit = np.matmul(x, b) + c
#         # # reshape data for prediction
#         # A.predict(data['Volatility_Time'][13:16].values.reshape(1, -1))
#         # #
#         prediction=[]
#         x = [i for i in range(n)]
#         for initial in range(warmup_period, len(data['Volatility_Time'])-n-1):
#             for i in range(n):
#                 x[i] = data['Volatility_Time'][i:(initial + i + 2)]
#                 # x[i] = x[i].reshape(len(x[i]), 1)
#             xstacked = np.column_stack(x)
#             # x[i] = data[(n-i-1):(-i-1)]
#             y = data['Volatility_Time'][n:n+initial+2]
#             A = lr()
#             A.fit(xstacked, y)
#             b = [A.coef_[i] for i in range(n)]
#             c = A.intercept_
#             # yfit = np.matmul(x, b) + c
#             # # reshape data for prediction
#             # A.predict(data['Volatility_Time'][13:16].values.reshape(1, -1))
#
#             if n == 1:
#                 mat1 = np.transpose([c, b[0]])
#                 mat2 = [1, y[y.last_valid_index()]]
#                 prediction.append(np.matmul(mat1, mat2))
#             if n == 3:
#                 mat1 = np.transpose([c, b[0], b[1], b[2]])
#                 mat2 = [1, x[1][x[1].last_valid_index()], x[2][x[2].last_valid_index()], y[y.last_valid_index()]]
#                 prediction.append(np.matmul(mat1, mat2))
#             if n == 5:
#                 mat1 = np.transpose([c, b[0], b[1], b[2], b[3], b[4]])
#                 mat2 = [1, x[1][x[1].last_valid_index()], x[2][x[2].last_valid_index()], x[3][x[3].last_valid_index()],
#                         x[4][x[4].last_valid_index()], y[y.last_valid_index()]]
#                 prediction.append(np.matmul(mat1, mat2))
#             if n == 10:
#                 mat1 = np.transpose([c, b[0], b[1], b[2], b[3], b[4], b[5],b[6],b[7],b[8],b[9]])
#                 mat2 = [1, x[1][x[1].last_valid_index()], x[2][x[2].last_valid_index()], x[3][x[3].last_valid_index()],
#                         x[4][x[4].last_valid_index()], x[5][x[5].last_valid_index()],x[6][x[6].last_valid_index()],
#                         x[7][x[7].last_valid_index()],x[8][x[8].last_valid_index()],x[9][x[9].last_valid_index()],
#                         y[y.last_valid_index()]]
#                 prediction.append(np.matmul(mat1, mat2))
#
#         y = y[(y.first_valid_index()+warmup_period-n-1):]
#         prediction = np.array(prediction)
#
#         Performance_ = PerformanceMeasure()
#         MSE = Performance_.mean_se(observed=y, prediction=prediction)
#         # QL = Performance_.quasi_likelihood(observed=y, prediction=prediction)
#         QL = Performance_.quasi_likelihood(observed=y*1000, prediction=prediction*1000)
#
#         dates = data['Date'][n:]
#         label=str(filename)+" "+str(stringinput)+" Linear Regression: "+str(n) + " Past Vol SE "
#         SE(y, prediction, dates,function_method=label)
#
#         # plt.title(str(filename)+" "+str(stringinput)+" Linear Regression: "+str(n) + " Past Vol SE ")
#
#         return MSE, QL, b, c
#

