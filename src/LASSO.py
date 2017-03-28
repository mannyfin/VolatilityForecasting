from VAR2 import *
import numpy as np
from sklearn.linear_model import Lasso
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt

# def lasso_regression(y, xw, alpha = 0.1, stringin):
def lasso_regression(xmat, ymat, p, dates, alpha):
    # q = 9
    # p = 3
    # n = p * 100
    observed = ymat
    MSE_list=[]

    for col in ymat.keys():
        MSE_list = []
        for i_alpha in alpha:
            observed = ymat[col]
            lassoreg = Lasso(alpha=i_alpha)
            lassoreg.fit(xmat, observed)  # pass in a single col
            # prediction = lassoreg.predict(observed.iloc[-1].values.reshape(1, -1))
            # prediction = lassoreg.predict(ymat.iloc[-1].values.reshape(1, -1))
            prediction = lassoreg.predict(xmat.iloc[-1].values.reshape(1,-1))

            # prediction = pd.DataFrame(y_pred,columns=ymat.keys())
            # observed = ymat

            # Performance_ = PerformanceMeasure()
            # MSE = Performance_.mean_se(observed=observed, prediction=prediction)

            MSE = np.sum(np.longdouble(np.mean(np.square(np.subtract(observed.iloc[-1], prediction)))))
            # mse_sum = pd.Series(np.sum(MSE), index=['SumMSE'])

            MSE_list.append(MSE)

        MSE_list = pd.DataFrame(MSE_list)
        plt.plot(alpha, MSE_list)
        plt.xlabel('lamda')
        plt.ylabel('sum MSE')
        plt.title('LASSO training for '+str(col)+' for p = '+str(p))
        # the line below gives the first index of the lamda with the minimum MSE' lamda
        lamda_p=alpha[MSE_list.idxmin()[0]]
        print(lamda_p)
        plt.show()
        print("hi")

    return lamda_p
    # lasso: do mse and ql
