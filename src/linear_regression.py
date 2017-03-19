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

        # data = data.reset_index(range(len(data)))
        LogVol = np.log(data['Volatility_Time'])
        PredictedLogVol=[]
        x = [i for i in range(n)]
        for initial in range(warmup_period, len(LogVol)):
        # for initial in range(warmup_period, len(LogVol)-1):
        # for initial in range(warmup_period, len(LogVol)-n):
            for i in range(n):
                x[i] = LogVol[i:(initial +i-n-1)]
                # x[i] = LogVol[i:(initial +i-n+1)]
                # x[i] = LogVol[i:(initial +i)]

            xstacked = np.column_stack(x)

            y = LogVol[n:initial-1]
            # y = LogVol[n:initial+1]
            # y = LogVol[n:n+initial]
            A = lr()
            A.fit(xstacked, y)
            b = [A.coef_[i] for i in range(n)]
            c = A.intercept_

            # TODO check that LR is correct..that we are predicting out of sample
            # TODO add functional to choose log instead of hard coding log in the code here
            # # reshape data for prediction
            PredictedLogVol.append(A.predict(LogVol[initial-n : initial].values.reshape(1, -1))[0])
            # PredictedLogVol.append(A.predict(LogVol[initial-n+1 : initial+1].values.reshape(1, -1))[0])
            # PredictedLogVol.append(A.predict(LogVol[initial+1:n+initial+1].values.reshape(1, -1))[0])


        y = data.Volatility_Time[warmup_period:]
        # y = data.Volatility_Time[warmup_period+1:]
        # y = data.Volatility_Time[warmup_period+n:]
        PredictedVol = pd.Series(np.exp(PredictedLogVol))
        Performance_ = PerformanceMeasure()
        MSE = Performance_.mean_se(observed=y, prediction=PredictedVol)
        QL = Performance_.quasi_likelihood(observed=y, prediction=PredictedVol)

        dates = data['Date'][n:]
        label=str(filename)+" "+str(stringinput)+" Linear Regression: "+str(n) + " Past Vol SE "
        SE(y, PredictedVol, dates,function_method=label)

        # plt.title(str(filename)+" "+str(stringinput)+" Linear Regression: "+str(n) + " Past Vol SE ")

        return MSE, QL, b, c


        # data = data.reset_index(range(len(data)))
        # LogVol = np.log(data['Volatility_Time'])
        # prediction=[]
        # x = [i for i in range(n)]
        # for initial in range(warmup_period, len(data['Volatility_Time'])-n):
        #     for i in range(n):
        #         x[i] = data['Volatility_Time'][i:(initial +i)]
        #
        #     xstacked = np.column_stack(x)
        #
        #     y = data['Volatility_Time'][n:n+initial]
        #     A = lr()
        #     A.fit(xstacked, y)
        #     b = [A.coef_[i] for i in range(n)]
        #     c = A.intercept_
        #
        #     # # reshape data for prediction
        #     prediction.append(A.predict(data.Volatility_Time[initial:n+initial].values.reshape(1, -1))[0])
        #
        #
        # y = data.Volatility_Time[warmup_period+n:]
        # prediction = pd.Series(prediction)
        # Performance_ = PerformanceMeasure()
        # MSE = Performance_.mean_se(observed=y, prediction=prediction)
        # # QL = Performance_.quasi_likelihood(observed=y, prediction=prediction)
        # QL = Performance_.quasi_likelihood(observed=y, prediction=prediction)
        #
        # dates = data['Date'][n:]
        # label=str(filename)+" "+str(stringinput)+" Linear Regression: "+str(n) + " Past Vol SE "
        # SE(y, prediction, dates,function_method=label)
        #
        # # plt.title(str(filename)+" "+str(stringinput)+" Linear Regression: "+str(n) + " Past Vol SE ")
        #
        # return MSE, QL, b, c





