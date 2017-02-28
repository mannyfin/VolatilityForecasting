import matplotlib.pyplot as plt
import numpy as np

# from sklearn.metrics import mean_squared_error as mse
from SEplot import se_plot as SE
from sklearn.linear_model import LinearRegression as lr
from Performance_Measure import *


class LinRegression:

    def lin_reg(data, n, filename, stringinput):

        # data = np.asarray(data)

        x = [i for i in range(n)]
        for i in range(n):
            x[i] = data['Volatility_Time'][(n - i - 1):(-i - 1)]
            # x[i] = x[i].reshape(len(x[i]), 1)
        x = np.column_stack(x)
        # x[i] = data[(n-i-1):(-i-1)]
        y = data['Volatility_Time'][n:]
        A = lr()
        A.fit(x, y)
        b = [A.coef_[i] for i in range(n)]
        c = A.intercept_
        yfit = np.matmul(x, b) + c

        Performance_ = PerformanceMeasure()
        MSE = Performance_.mean_se(observed=y, prediction=yfit)
        QL = Performance_.quasi_likelihood(observed=y, prediction=yfit)

        dates = data['Date'][n:]
        SE(y, yfit, dates)

        plt.title(str(filename)+" "+str(stringinput)+" Linear Regression: "+str(n) + " Past Vol SE ")

        return MSE, QL, b, c


