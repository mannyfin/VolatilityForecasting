

import pandas as pd
import os
import numpy as np
from linear_regression_combined import lin_reg_comb as lrc

os.getcwd()
os.chdir('..')
os.chdir('Data')
train_set = pd.read_csv('train_set_comb.csv')
test_set = pd.read_csv('test_set_comb.csv')
os.chdir('..')
os.chdir('Ridge')
# # separate date out and remove date from train_set
# Date = train_set.Date
# train_set.drop('Date', axis=1, inplace=True)
# # take log vol
# train_set = train_set.apply(np.log)
# # split up data
# blah = train_set.T
# blah = blah.unstack()
warmup_period = 300
n = 5
MSE, QL, ln_SE, b, c = lrc(train_set, n=n, warmup_period=300)



print('hi')
print('hi')