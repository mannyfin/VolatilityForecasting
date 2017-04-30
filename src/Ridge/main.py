
import pandas as pd
import numpy as np
from preprocess_data import preprocess
from PaPmain import PaP
from LRmain import LR
from RRmain import RR
from BRRmain import BRR
import os


# filenames = ['AUDUSD.csv']
filenames = ['AUDUSD.csv', 'CADUSD.csv',  'CHFUSD.csv', 'EURUSD.csv', 'GBPUSD.csv', 'NOKUSD.csv', 'NZDUSD.csv']
filenames_nocsv = [name.replace(".csv", "") for name in filenames]

dailyvol_zeroes = pd.DataFrame()

# vars
warmup_period = 300
lr_optimal_n_list_benchmark = [3,3,3,3,3,3,3]
# lr_optimal_n_list_benchmark = [9,7,7,7,7,7,10]
n_seq = np.arange(1, 16, 1)
lamda_seq = np.exp(np.arange(-1, 3.1, 0.2))  # for RR1 and RR2
# for BRR1
param_range1 = np.exp(np.arange(-5, 3, 1))
param_range2 = np.exp(np.arange(-5, 3, 1))
param_range3 = np.exp(np.arange(-5, 3, 1))
param_range4 = np.exp(np.arange(-5, 3, 1))

dictlist = dict()

for count, name in enumerate(filenames):
    train_set, test_set, name = preprocess(name)

    """
            PastAsPresent
    """
    # pap_mse_list, pap_ql_list = PaP(test_set, name, dictlist)
    dictlist = PaP(test_set=test_set, name=name, count=count, dictlist=dictlist)

    """
            Linear Regression
    """
    # lr_mse_list, lr_ql_list, lr_optimal_n_list = LR(train_set, test_set, warmup_period, name,n_seq)
    dictlist = LR(train_set=train_set, test_set=test_set, warmup_period=warmup_period, name=name, count=count,
                   n_seq=n_seq, dictlist=dictlist)

    """
            Ridge Regression
    """
    n = 5

    dictlist = RR(train_set=train_set, test_set=test_set, warmup_period=warmup_period, name=name, count=count, n_seq=n,
                  lamda_seq=lamda_seq, lr_optimal_n_list_benchmark=lr_optimal_n_list_benchmark, dictlist=dictlist)

    """
            Bayesian Ridge Regression
    """
    n = 5

    dictlist = BRR(train_set, test_set, warmup_period=warmup_period, name=name, count=count, n_seq=n, dictlist=dictlist,
                   param_range=(param_range1, param_range2, param_range3, param_range4))

    """
            Kernel Ridge Regression
    """



# Post_processing
os.chdir('..')
# TODO add decorator to functions above to see if they were actually called
"""
        Bayesian Ridge Regression
"""
brr_test_restuls_df = pd.DataFrame({"BRR MSE":dictlist['BRR']['brr1_mse_list'],
                                     "BRR QL": dictlist['BRR']['brr1_ql_list'],
                                   "Optimal alpha1": dictlist['BRR']['brr_optimal_log_alpha1_list'],
                                    "Optimal alpha2": dictlist['BRR']['brr_optimal_log_alpha2_list'],
                                    "Optimal lambda1": dictlist['BRR']['brr_optimal_log_lambda1_list'],
                                    "Optimal lambda2": dictlist['BRR']['brr_optimal_log_lambda2_list']})

brr_test_restuls_df = brr_test_restuls_df.set_index(np.transpose(filenames_nocsv), drop=True)
brr_test_restuls_df.to_csv('BRR_test_MSE_QL.csv')

# lnSE to csv
brr1_lnSE = pd.DataFrame(dictlist['BRR']['brr1_lnSE_list']).T
brr1_lnSE.to_csv('BRR1_lnSE_list.csv')

# Predicted vol for each file to csv
brr1_PredVol = pd.DataFrame(dictlist['BRR']['brr1_PredVol_list']).T
brr1_PredVol.to_csv('BRR1_predictedVol.csv')

"""
        Kernel Ridge Regression
"""

print("Hi")
