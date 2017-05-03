
import pandas as pd
import numpy as np
from preprocess_data import preprocess

from post_processing import *
import os
from Combined_methods import *

filenames = ['AUDUSD.csv', 'CADUSD.csv',  'CHFUSD.csv', 'EURUSD.csv', 'GBPUSD.csv', 'NOKUSD.csv', 'NZDUSD.csv']



names = [name.split('.')[0] for name in filenames]

dailyvol_zeroes = pd.DataFrame()

# variables

warmup_period = 300
lr_optimal_n_list_benchmark = [3,3,3,3,3,3,3]
lr_optimal_n_list_benchmark = 3

n_seq = np.arange(5, 8, 1)
# for RR
lamda_seq =  np.exp(np.arange(-1, 0, 0.5))
# for BRR1
param_range1 = np.exp(np.arange(-5, 3, 1))
param_range2 = np.exp(np.arange(-5, 3, 1))
param_range3 = np.exp(np.arange(-5, 3, 1))
param_range4 = np.exp(np.arange(-5, 3, 1))

# for KRR
# alpha must be a list, coef0 must be a list
kernels = ['linear', 'polynomial', 'rbf', 'laplacian']
krr_alpha = [0.5, 1, 1.5]
krr_coef0 = [0.5, 1, 1.5]
krr_degree = [1, 3]

dictlist = dict()

os.getcwd()
os.chdir('..')
os.chdir('Data')
train_set = pd.read_csv('train_set_comb.csv')
test_set = pd.read_csv('test_set_comb.csv')
os.chdir('..')
os.chdir('Ridge')



"""
        PastAsPresent
"""
dictlist = PaP(test_set=test_set, train_set=train_set, name='Combined', dictlist=dictlist, filenames=names)


"""
        Linear Regression
"""
dictlist = LR(train_set=train_set, test_set=test_set, warmup_period=warmup_period, name='Combined',
               n_seq=n_seq, dictlist=dictlist, filenames=names)


"""
        Ridge Regression
"""
n = 5

dictlist = RR(train_set=train_set, test_set=test_set, warmup_period=warmup_period, name='Combined', n_seq=n,
              lamda_seq=lamda_seq, lr_optimal_n_list_benchmark=lr_optimal_n_list_benchmark, dictlist=dictlist, filenames=filenames)


"""
        Bayesian Ridge Regression
"""
n = 5

dictlist = BRR(train_set, test_set, warmup_period=warmup_period, name='Combined', n_seq=n, dictlist=dictlist,
               param_range=(param_range1, param_range2, param_range3, param_range4), filenames=filenames)


"""
        Kernel Ridge Regression
"""
n = 5
dictlist, kernels = KRR(train_set, test_set, warmup_period=warmup_period, name='Combined', n_seq=n, kernels=kernels,
                        dictlist=dictlist, param_range=(krr_alpha, krr_coef0, krr_degree), filenames=filenames)

# TODO add decorator to functions above to see if they were actually called
# TODO fix directory issue in post_processing section of code below.

"Post_processing"
# manually run for combined methods in KRR
# there may be bugs for these. be careful

os.chdir('..')
PaP_post_process(dictlist, names)

os.chdir('..')
BRR_post_process(dictlist, names)


os.chdir('..')
KRR_post_process(dictlist, names, kernels)

print("Complete")

