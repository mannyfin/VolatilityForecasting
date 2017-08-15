
import pandas as pd
import numpy as np
from preprocess_data import preprocess
from PaPmain import PaP
from LRmain import LR
from RRmain import RR
from BRRmain import BRR
from KRRmain import KRR
from post_processing import *
import os


# filenames = ['AUDUSD.csv']
filenames = ['AUDUSD.csv', 'CADUSD.csv',  'CHFUSD.csv', 'EURUSD.csv', 'GBPUSD.csv', 'NOKUSD.csv', 'NZDUSD.csv']

# filenames_nocsv = [name.replace(".csv", "") for name in filenames]

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

# for KRR
# alpha must be a list, coef0 must be a list
kernels = ['linear', 'polynomial', 'rbf', 'laplacian']
krr_alpha = [1,2]# np.exp(np.arange(0, 3, 1))
krr_coef0 = [1,2] #np.exp(np.arange(0, 3, 1))
krr_degree = [2]

dictlist = dict()
names = []
for count, name in enumerate(filenames):
    train_set, test_set, name = preprocess(name)
    names.append(name)

    """
            PastAsPresent
    """
    # pap_mse_list, pap_ql_list = PaP(test_set, name, dictlist)
    dictlist = PaP(test_set=test_set, train_set=train_set, name=name, count=count, dictlist=dictlist)


    # """
    # #         Linear Regression
    # # """
    # # lr_mse_list, lr_ql_list, lr_optimal_n_list = LR(train_set, test_set, warmup_period, name,n_seq)
    # dictlist = LR(train_set=train_set, test_set=test_set, warmup_period=warmup_period, name=name, count=count,
    #                n_seq=n_seq, dictlist=dictlist)
    #

    # """
    #         Ridge Regression
    # """
    # n = 5
    #
    # dictlist = RR(train_set=train_set, test_set=test_set, warmup_period=warmup_period, name=name, count=count, n_seq=n,
    #               lamda_seq=lamda_seq, lr_optimal_n_list_benchmark=lr_optimal_n_list_benchmark, dictlist=dictlist)
    #

    # """
    #         Bayesian Ridge Regression
    # """
    # n = 5
    #
    # dictlist = BRR(train_set, test_set, warmup_period=warmup_period, name=name, count=count, n_seq=n, dictlist=dictlist,
    #                param_range=(param_range1, param_range2, param_range3, param_range4))


    # """
    #         Kernel Ridge Regression
    # """
    # n = 5
    # dictlist, kernels = KRR(train_set, test_set, warmup_period=warmup_period, name=name, count=count, n_seq=n, kernels=kernels,
    #                         dictlist=dictlist, param_range=(krr_alpha, krr_coef0, krr_degree))

# TODO add decorator to functions above to see if they were actually called
# TODO fix directory issue in post_processing section of code below.

"Post_processing"

# os.chdir('..')
PaP_post_process(dictlist, names)

# os.chdir('..')
# BRR_post_process(dictlist, names)

os.chdir('..')
# KRR_post_process(dictlist, names, kernels)

print("Complete")
