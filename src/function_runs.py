from PastAsPresent import *
from linear_regression import *
# from arch_q_model import ArchModelQ as am
from garch_pq_model import GarchModel as gm
import numpy as np


class FunctionCalls(object):

    input_data = None
    tnplus1 = 0
    lr = 0
    arch = 0
    garch11 = 0

    def __init__(self):
        pass

    def function_runs(self,stringinput, input_data, tnplus1, lr, arch, garch11):
        output = {}

        """tnplus1"""
        try:
            if tnplus1 == 1:
                part1 = PastAsPresent.tn_pred_tn_plus_1(input_data)
                output['PastAsPresent'] = part1
            elif tnplus1 == 0:
                pass
        except ValueError:
            print("Error: Make sure you pass in 1 or  0 for arg in tnplus1... ")

        """Linear Regression"""
        try:
            # not the best exception handling here...
            if len(lr)>= 1 & isinstance(lr, list):
                for count, elem in enumerate(lr):
                    LRmethod = LinRegression.lin_reg(input_data, elem)
                    output['LinearRegression' + str(elem)] = LRmethod[0:2]
            else:
                pass

        except TypeError:
            print("Error: Please pass an array of ints...")
        try:
            if len(arch) == 4:
                ARCH = gm.arch_q_mse(data=input_data,Timedt=stringinput, ret=arch[0], q=arch[1], lags=arch[2], initial=arch[3])
                output['ARCH'] = ARCH
        except TypeError:
            print("Error: ARCH, make sure all the params are filled")

        try:
            if len(garch11) == 5:
                GARCH = gm.garch_pq_mse(data=input_data,Timedt=stringinput, ret=garch11[0], p=garch11[1], q=garch11[2], lags=garch11[3],
                                        initial=garch11[4])
                output['GARCH'] = GARCH
        except TypeError:
            print("Error: GARCH, make sure all the params are filled")

        return output
