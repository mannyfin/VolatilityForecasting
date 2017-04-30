

import pandas as pd
import linear_regression as lr
import read_in_files as rd
import Volatility as vol
import split_data as sd
from PastAsPresent import PastAsPresent as pap
import RidgeRegression as rr
import BayesianRegression as brr
import KernelRidgeRegression as krr
import matplotlib.pyplot as plt
import numpy as np
from preprocess_data import preprocess
from PaPmain import PaP
from LRmain import LR
from RRmain import RR
from BRRmain import BRR
import makedirs
from append_lists import initialize_lists, append_outputs
from dictstruct import Struct
import os

# filenames = ['AUDUSD.csv']
filenames = ['AUDUSD.csv', 'CADUSD.csv',  'CHFUSD.csv', 'EURUSD.csv', 'GBPUSD.csv', 'NOKUSD.csv', 'NZDUSD.csv']
filenames_nocsv = [name.replace(".csv", "") for name in filenames]

dailyvol_zeroes= pd.DataFrame()

# vars
warmup_period = 300
lr_optimal_n_list_benchmark = [3,3,3,3,3,3,3]
# lr_optimal_n_list_benchmark = [9,7,7,7,7,7,10]
n_seq = np.arange(1, 16, 1)
lamda_seq = np.exp(np.arange(-1, 3.1, 0.2))  # for RR1 and RR2

# initialize_lists(pap=True)
# dictlist = Struct(dict())
dictlist = dict()

for count, name in enumerate(filenames):
    train_set, test_set, name = preprocess(name)
    #
    # """
    #         PastAsPresent
    # """
    # # pap_mse_list, pap_ql_list = PaP(test_set, name, dictlist)
    # dictlist = PaP(test_set=test_set, name=name, dictlist=dictlist)
    #
    # """
    #         Linear Regression
    # """
    # # lr_mse_list, lr_ql_list, lr_optimal_n_list = LR(train_set, test_set, warmup_period, name,n_seq)
    # dictlist = LR(train_set=train_set, test_set=test_set, warmup_period=warmup_period, name=name, n_seq=n_seq,
    #               dictlist=dictlist)
    #
    # """
    #         Ridge Regression
    # """
    # n = 5
    #
    # dictlist = RR(train_set=train_set, test_set=test_set, warmup_period=warmup_period, name=name,n_seq=n,
    #               lamda_seq=lamda_seq, lr_optimal_n_list_benchmark=lr_optimal_n_list_benchmark, count=count,
    #               dictlist=dictlist)

    """
            Bayesian Ridge Regression
    """
    n = 5

    dictlist = BRR(train_set, test_set, warmup_period, name, n, dictlist)



