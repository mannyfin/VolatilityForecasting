"""for HW2.1 main.py is used for running VAR. main 

use this file to run KNN and KNN with time component. use main.py to run VAR"""


#  if you dont have this installed then here's how to install the module:
#  conda install -c https://conda.binstar.org/bashtage arch

from read_in_files import read_in_files
from NumDaysWeeksMonths import NumDaysWeeksMonths
from Volatility import *
from linear_regression import *
import matplotlib.pyplot as plt
from PastAsPresent import *
# from VAR import *
from tablegen import tablegen
from garch_pq_model import GarchModel as gm
# from arch_q_model import ArchModelQ as am
import numpy as np
import pandas as pd
from KNN import KNN
from function_runs import *
from VAR2 import *
import matplotlib.backends.backend_pdf


filenames = ['AUDUSD.csv', 'CADUSD.csv',  'CHFUSD.csv', 'EURUSD.csv', 'GBPUSD.csv', 'NOKUSD.csv', 'NZDUSD.csv']
# filenames = ['SEKUSD.csv','CADUSD.csv',  'CHFUSD.csv',]

dailyvol_zeroes= pd.DataFrame()
weeklyvol_zeroes= pd.DataFrame()
monthlyvol_zeroes= pd.DataFrame()
dailyret_zeroes= pd.DataFrame()
weeklyret_zeroes= pd.DataFrame()
monthlyret_zeroes = pd.DataFrame()
Daily_list = list()
namelist = list()
for count, name in enumerate(filenames):
    #  reads in the files and puts them into dataframes, returns a dataframe called df
    df, df_single_day, df_single_week, df_single_month = read_in_files(name)

    name = name.split('.')[0]
    namelist.append(name)
    print("Running file: " + str(name))
    warmup_period = 400
    daily_vol_result, daily_ret, daily_vol_zeroes, daily_ret_zeroes = time_vol_calc(df_single_day)

    # be careful of the underscore, _
    dailyvol_zeroes = pd.concat([dailyvol_zeroes, daily_vol_zeroes], axis=1)

    # dailyvol_zeroes = pd.concat([dailyvol_zeroes, daily_vol_zeroes['Volatility_Time']], axis=1)

    dailyvol_zeroes.rename(columns={'Volatility_Time': name}, inplace=True)

    dailyret_zeroes = pd.concat([dailyret_zeroes, daily_ret_zeroes], axis=1)

    dailyret_zeroes.rename(columns={'Return_Time': name}, inplace=True)

"remove the first month because it is incomplete"
print("hi")
# does not have zeroes
daily_vol_combined = dailyvol_zeroes[(dailyvol_zeroes != 0).all(1)]

dates = daily_vol_combined.Date.loc[:, ~daily_vol_combined.Date.columns.duplicated()].reset_index()
# drop duplicate columns
daily_vol_combined.drop('Date', axis=1, inplace=True)
daily_vol_combined=daily_vol_combined.apply(pd.to_numeric)

"""I am referencing the $Time$vol_zeroes variable in the lines below because there are (or could be) days where the ret
  is zero"""
daily_ret_combined = dailyret_zeroes[(dailyvol_zeroes != 0).all(1)]

# We need to reset index
daily_vol_combined.reset_index(drop=True, inplace=True)
daily_ret_combined.reset_index(drop=True, inplace=True)

fc = FunctionCalls()

VAR_test = fc.function_runs(dates=dates, filename='Combined Curr.', stringinput='Daily', warmup=909, input_data=np.log(daily_vol_combined), var_lag=[1, 2, 3])

# """Output multiple plots into a pdf file"""
# pdf = matplotlib.backends.backend_pdf.PdfPages(name+".pdf")
# for fig in range(1, 3*count+3+1):
#     pdf.savefig( fig, dpi=1200 )
# pdf.close()
#
#
plt.show()
print("Complete")