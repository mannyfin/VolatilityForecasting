import pandas as pd
import numpy as np
from Performance_Measure import *
from SEplot import se_plot as SE
import matplotlib.pyplot as plt
from LASSO import lasso_regression

class VAR(object):
    """
    VAR forecaster

    """

    def __init__(self, p, combined_vol, warmup_period):
        assert isinstance(combined_vol, pd.core.frame.DataFrame)
        self.p = p
        self.combined_vol = combined_vol
        self.warmup_period = warmup_period

    def VAR_calc(self, Timedt, dates, filename, doLASSO_only=False):
        # provides the whole x matrix.
        self.xmat = pd.DataFrame([sum([self.combined_vol[currency].loc[i + self.p - 1:i:-1].as_matrix().tolist()
                                  for currency in self.combined_vol.keys()], [])
                                  for i in range(len(self.combined_vol) - self.p)])

        # provides the whole y matrix
        self.ymat = self.combined_vol[self.p:]
        """
        initial xmat, ymat
        self.xmat[:self.warmup_period]

        feel free to use log of the vols, or whatever you'd like as an input. Doesn't need to be defined inside the fcn.
        ymat = daily_vol_combined[p:warmup_period + p]
        test = xmat[:warmup_period]

        check: ymat[-4:] vs xmat: test[-3:]

        Ex.p = 3 and warmup = 100 here...
        ymat[-4:]
                Out[236]:
                       SEKUSD    CADUSD    CHFUSD
                99   0.207160  0.132623  0.180368
                100  0.193095  0.115839  0.146339
                101  0.202393  0.119725  0.158681
                102  0.185685  0.113315  0.147309

        test[-3:]
                Out[238]:
                           0         1         2         3         4         5         6  \
                97  0.207160  0.217591  0.262496  0.132623  0.157432  0.204130  0.180368
                98  0.193095  0.207160  0.217591  0.115839  0.132623  0.157432  0.146339
                99  0.202393  0.193095  0.207160  0.119725  0.115839  0.132623  0.158681
                     7         8
                97  0.182417  0.224175
                98  0.180368  0.182417
                99  0.146339  0.180368

        # Calculate beta
        # beta = (X_T * X)^-1 * ( X_T * Y)
        beta = np.matmul(np.linalg.pinv(self.xmat.T.dot(self.xmat)), np.matmul(self.xmat.T, self.ymat))

        Calculate y_predicted:

        y_predicted = X_T1*beta_T1,fit

            where y_predicted = y_T1+1

        the line below is wrong because it uses X_T1-1 instead of X_T1

        y_prediction = np.matmul(self.test[-1:], beta)

        Here is the last row of ymat. i.e. y_T1
            ymat[-1:]
            Out[240]:
                   SEKUSD    CADUSD    CHFUSD
            102  0.185685  0.113315  0.147309

        We instead index into the row after the last row of xmat_var using xmat (the complete one)

         This is incorrect:
            test[-1:]
            Out[239]:
                       0         1        2         3         4         5         6  \
            99  0.202393  0.193095  0.20716  0.119725  0.115839  0.132623  0.158681
                       7         8
            99  0.146339  0.180368

         This is correct:
            xmat[len(test):len(test)+1]
            Out[230]:
                        0         1         2         3         4         5         6  \
            100  0.185685  0.202393  0.193095  0.113315  0.119725  0.115839  0.147309
                        7         8
            100  0.158681  0.146339

        Notice columns, 0, 3, 6 are the elements in ymat[-1:] (i.e. y_T1).
        This means that xmat[len(test):len(test)+1] is X_T1

        We use this to calculate y_predicted: (i.e. y_T1+1):

            y_predicted = X_T1*beta_T1
len(self.xmat)-self.warmup_period)
        """
        if doLASSO_only==False:
            beta = []
            prediction=[]
            for iteration in range(len(self.ymat)-self.warmup_period):

                # X goes from 0 to warmup_period (T1-1). Ex. for p=3 and warmup=100,
                # x index goes from 0 to 99, & col=3*numfiles
                xmat_var = self.xmat[:(self.warmup_period+ iteration) ]

                # Y goes from p to the warmup period+p. Ex. for p = 3 and warmup = 100x y index goes from 3 to 102 inclusive
                ymat_var = self.combined_vol[self.p:(self.warmup_period + self.p + iteration)]

                # We can ravel the betas below to stack them row by row if we want to use them later for a pandas DataFrame
                # The ravel would have to be done after the prediction.append() line.
                beta.append(np.matmul(np.linalg.pinv(xmat_var.T.dot(xmat_var)), np.matmul(xmat_var.T, ymat_var)))

                # the x used here is the row after the warmup period, T1. i.e. if xmat_var is from 0:99 inclusive, then
                # value passed for x is row 100
                prediction.append(np.matmul(self.xmat[len(xmat_var):len(xmat_var) + 1], beta[-1])[0].tolist())
            prediction = pd.DataFrame(prediction, columns=self.combined_vol.keys())

            # observed: ex.For the case of p = 3 is from index 103:1299 inclusive, 1197 elements total for warmup_period=100
            observed = self.ymat[self.warmup_period:]
            # now calculate MSE, QL and so forth
            Performance_ = PerformanceMeasure()
            MSE = Performance_.mean_se(observed=observed, prediction=prediction)
            QL = Performance_.quasi_likelihood(observed=observed, prediction=prediction)

            """ return a plot of the Squared error"""
            label = str(filename) + " " + str(Timedt) + " SE (" + str(self.p) + ") VAR Volatility"
            SE(observed, prediction, dates.iloc[(self.warmup_period+self.p):], function_method=label)
            plt.title('VAR for p = '+str(self.p))

        elif doLASSO_only==True:
            print("Performing LASSO regression")
            blah=lasso_regression(self.xmat,self.ymat, self.p, dates, alpha=np.linspace(0.0000000001,0.0000005,1e4))
            MSE=[]
            QL=[]

        return MSE, QL