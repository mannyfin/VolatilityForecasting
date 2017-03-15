from PastAsPresent import *
from linear_regression import *
from garch_pq_model import GarchModel as gm
from KNN import KNN
import numpy as np
import pandas as pd
from res2df_list import *


class FunctionCalls(object):

    input_data = None
    tnplus1 = 0
    lr = 0
    arch = 0
    garchpq = 0

    def __init__(self):
        pass

    def function_runs(self, filename, stringinput, warmup,input_data, tnplus1=None, lr=None, arch=None, garchpq=None, k_nn=None):
        output = list()

        """tnplus1"""
        try:
            if tnplus1 is not None :
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
                print("None linear regressor")
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
        try:
            #
            if arch is None:
                print("None for arch")
            elif len(arch) == 3:
                ARCH = gm.arch_q_mse(data=input_data, Timedt=stringinput, ret=arch[0], q=arch[1], lags=arch[2],
                                     warmup_period=warmup, filename=filename)
                # output['ARCH'] = ARCH

                output = result_to_df_list(list_name=output, method_result=ARCH,
                                           index_value=['ARCH'], column_value=['MSE', 'QL'])


                print("Above is ARCH for " + str(stringinput))
        except TypeError:
            print("Error: ARCH, make sure all the params are filled")

        try:
            # 4 is the num of args to pass into the fcn
            if garchpq is None:
                print("None for garch")
            elif len(garchpq) == 4:
                GARCH = gm.garch_pq_mse(data=input_data, Timedt=stringinput, ret=garchpq[0], p=garchpq[1], q=garchpq[2],
                                        lags=garchpq[3], warmup_period=warmup, filename=filename)

                # output['GARCH'] = GARCH

                output = result_to_df_list(list_name=output, method_result=GARCH,
                                           index_value=['GARCH'], column_value=['MSE', 'QL'])

                print("Above is GARCH for " + str(stringinput))
        except TypeError:
            print("Error: GARCH, make sure all the params are filled")

        try:
            # 4 is the num of args to pass into the fcn
            if k_nn is None:
                print("None for KNN")
            elif len(k_nn) >= 1 & isinstance(k_nn, list):
                for count, elem in enumerate(k_nn):
                    KNNmethod = KNN(vol_data=input_data, k=elem, warmup=warmup, filename=filename, Timedt=stringinput)

                    # output['KNN_'+str(k_nn)] = KNNmethod

                    output = result_to_df_list(list_name=output, method_result=KNNmethod,
                                               index_value=['KNN_'+str(k_nn)], column_value=['MSE', 'QL'])

                print("Above is KNN for " +str(elem)+ " " + str(stringinput))
        except TypeError:
            print("Error: KNN, make sure all the params are filled")

        # concatenates the list of df's
        output = list_to_df(list_name=output)

        return output
