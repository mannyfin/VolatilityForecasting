from PastAsPresent import *
from linear_regression import *
from garch_pq_model import GarchModel as gm
import numpy as np


class FunctionCalls(object):

    input_data = None
    tnplus1 = 0
    lr = 0
    arch = 0
    garchpq = 0

    def __init__(self):
        pass

    def function_runs(self, filename, stringinput, warmup,input_data, tnplus1, lr, arch, garchpq):
        output = {}

        """tnplus1"""
        try:
            if tnplus1 == 1:
                part1 = PastAsPresent.tn_pred_tn_plus_1(data=input_data, filename=filename, stringinput=stringinput)
                output['PastAsPresent'] = part1
                print("Above is Past as present for " + str(stringinput))
            elif tnplus1 == 0:
                pass
        except ValueError:
            print("Error: Make sure you pass in 1 or  0 for arg in tnplus1... ")

        """Linear Regression"""
        try:
            # not the best exception handling here...
            if len(lr)>= 1 & isinstance(lr, list):
                for count, elem in enumerate(lr):
                    LRmethod = LinRegression.lin_reg(data=input_data, n=elem, filename=filename,
                                                     stringinput=stringinput, warmup_period=warmup)
                    output['LinearRegression_' + str(elem)] = LRmethod[0:2]
                    print("Above is LR for " +str(elem)+" "+ str(stringinput) +" Volatilities")
            else:
                pass

        except TypeError:
            print("Error: Please pass an array of ints...")
        try:
            #
            if len(arch) == 3:
                ARCH = gm.arch_q_mse(data=input_data, Timedt=stringinput, ret=arch[0], q=arch[1], lags=arch[2],
                                     warmup_period=warmup, filename=filename)
                output['ARCH'] = ARCH
                print("Above is ARCH for " + str(stringinput))
        except TypeError:
            print("Error: ARCH, make sure all the params are filled")

        try:
            # 4 is the num of args to pass into the fcn
            if len(garchpq) == 4:
                GARCH = gm.garch_pq_mse(data=input_data, Timedt=stringinput, ret=garchpq[0], p=garchpq[1], q=garchpq[2],
                                        lags=garchpq[3], warmup_period=warmup, filename=filename)
                output['GARCH'] = GARCH
                print("Above is GARCH for " + str(stringinput))
        except TypeError:
            print("Error: GARCH, make sure all the params are filled")

        return output
