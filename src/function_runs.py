from PastAsPresent import *
from linear_regression import *
from garch_pq_model import GarchModel as gm
# from KNN import KNN
import numpy as np
from KNN import KNN
import pandas as pd
from res2df_list import *
from VAR2 import *


class FunctionCalls(object):

    input_data = None
    tnplus1 = 0
    lr = 0
    arch = 0
    garchpq = 0

    def __init__(self):
        pass

    def function_runs(self,dates=None, filename=None, stringinput=None, warmup=None,input_data=None, tnplus1=None, lr=None, arch=None, garchpq=None, k_nn=None, var_lag=None, LASSO=False):
        output = list()

        """tnplus1"""
        try:
            if tnplus1 is None:
                print("Not running tnplus1")
            elif tnplus1 is not None :
                tnplus1_method = PastAsPresent.tn_pred_tn_plus_1(data=input_data, filename=filename, stringinput=stringinput)
                # output['PastAsPresent'] = part1


                output = result_to_df_list(list_name=output, method_result=tnplus1_method,
                                           index_value=['PastAsPresent'], column_value=['MSE', 'QL'])

                # output['PastAsPresent'] = tnplus1_method
                print("Above is Past as present for " + str(stringinput))

        except ValueError:
            print("Error: Make sure you pass in 1 or  0 for arg in tnplus1... ")

        """Linear Regression"""
        try:
            if lr is None:
                print("Not running linear regression")
            elif len(lr)>= 1 & isinstance(lr, list):
                for count, elem in enumerate(lr):
                    LRmethod = LinRegression.lin_reg(data=input_data, n=elem, filename=filename,
                                                     stringinput=stringinput, warmup_period=warmup)
                    # output['LinearRegression_' + str(elem)] = LRmethod[0:2]

                    output = result_to_df_list(list_name=output, method_result=LRmethod[0:2],
                                               index_value=['LinearRegression_' + str(elem)], column_value=['MSE', 'QL'])

                    # output['LinearRegression_' + str(elem)] = LRmethod[0:2]
                    print("Above is LR for " +str(elem)+" "+ str(stringinput) +" Volatilities")
            else:
                pass

        except TypeError:
            print("Error: Please pass an array of ints...")

        """ARCH """
        try:
            #
            if arch is None:
                print("Not running arch")
            elif len(arch) == 3:
                ARCH = gm.arch_q_mse(data=input_data, Timedt=stringinput, ret=arch[0], q=arch[1], lags=arch[2],
                                     warmup_period=warmup, filename=filename)
                # output['ARCH'] = ARCH

                output = result_to_df_list(list_name=output, method_result=ARCH,
                                           index_value=['ARCH'], column_value=['MSE', 'QL'])


                print("Above is ARCH for " + str(stringinput))
        except TypeError:
            print("Error: ARCH, make sure all the params are filled")

        """GARCH """
        try:
            # 4 is the num of args to pass into the fcn
            if garchpq is None:
                print("Not running garch")
            elif len(garchpq) == 4:
                GARCH = gm.garch_pq_mse(data=input_data, Timedt=stringinput, ret=garchpq[0], p=garchpq[1], q=garchpq[2],
                                        lags=garchpq[3], warmup_period=warmup, filename=filename)

                # output['GARCH'] = GARCH

                output = result_to_df_list(list_name=output, method_result=GARCH,
                                           index_value=['GARCH'], column_value=['MSE', 'QL'])

                print("Above is GARCH for " + str(stringinput))
        except TypeError:
            print("Error: GARCH, make sure all the params are filled")

        """KNN """
        try:
            # 4 is the num of args to pass into the fcn
            if k_nn is None:
                print("Not running KNN")
            elif len(k_nn) >= 1 & isinstance(k_nn, list):
                for count, elem in enumerate(k_nn):
                    KNNmethod = KNN(vol_data=input_data, k=elem, warmup=warmup, filename=filename, Timedt=stringinput)

                    # output['KNN_'+str(k_nn)] = KNNmethod

                    output = result_to_df_list(list_name=output, method_result=KNNmethod,
                                               index_value=['KNN_'+str(k_nn)], column_value=['MSE', 'QL'])

                    print("Above is KNN for " +str(elem)+ " " + str(stringinput))
        except TypeError:
            print("Error: KNN, make sure all the params are filled")

        """VAR """
        try:
            # 4 is the num of args to pass into the fcn
            if var_lag is None:
                print("Not running VAR")
            elif len(var_lag) >= 1 & isinstance(var_lag, list):
                for count, elem in enumerate(var_lag):
                    # KNNmethod = KNN(vol_data=input_data, k=elem, warmup=warmup, filename=filename, Timedt=stringinput)
                    VAR_q = VAR(p=elem, combined_vol=input_data, warmup_period=warmup)\
                                .VAR_calc(Timedt=stringinput, dates=dates, filename=filename)
                    # import matplotlib.pyplot as plt
                    # plt.show()
                    # the line below doesnt work at the moment...
                    test = pd.DataFrame.from_records(VAR_q[0]).transpose()
                    test.columns = ['MSE']
                    test=test.rename(index={'SumMSE': 'Sum'})
                    test1 = pd.DataFrame.from_records(VAR_q[1]).transpose()
                    test1.columns = ['QL']
                    test1=test1.rename(index={'SumQL': 'Sum'})
                    output = pd.concat([test, test1], axis=1)
                    writer = pd.ExcelWriter('VAR_output p='+str(elem)+'.xlsx')
                    output.to_excel(writer, 'VAR p =' + str(elem))
                    writer.save()

                    output = result_to_df_list(list_name=output, method_result=output,
                                               index_value=['VAR_p='+str(elem)], column_value=['MSE', 'QL'])

                    print("Above is VAR for p=" +str(elem)+ " " + str(stringinput))

        except TypeError:
            print("Error: VAR, make sure all the params are filled")
        # concatenates the list of df's
        output = list_to_df(list_name=output)

        return output
