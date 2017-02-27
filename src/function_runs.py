from PastAsPresent import *
from linear_regression import *
# from arch_q_model import ArchModelQ as am
from garch_pq_model import GarchModel as gm


class FunctionCalls(object):

    input = None
    tnplus1 = 0
    lr = 0
    arch = 0
    garch11 = 0

    def __init__(self):
        pass

    def function_runs(self,stringinput, input, tnplus1, lr, arch, garch11):
        output = {}

        """tnplus1"""

        try:
            if tnplus1 ==1:
                output['PastAsPresent'] = PastAsPresent.tn_pred_tn_plus_1(input)
            else:
                pass
        except(ValueError):
            print("Error: Make sure you pass in 1 or  0 for arg in tnplus1... ")
        """Linear Regression"""
        try:
            # not the best exception handling here...
            if len(lr)>= 1 & isinstance(lr, list):
                for count, elem in enumerate(lr):
                    LRmethod = LinRegression.lin_reg(input, elem)
                    output['LinearRegression' + str(elem)] = LRmethod[0:2]
            else:
                pass

        except(ValueError):
            print("Error: Please pass an array of ints...")

        # if len(arch) ==3:
        #     am.arch_q_mse(daily_vol_result, np.array(daily_ret['Return_Daily']), 1, 1)


        return output