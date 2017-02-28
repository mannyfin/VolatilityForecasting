# This main file is a sketch of the overall program
#  if you dont have this installed then here's how to install the module:
#  conda install -c https://conda.binstar.org/bashtage arch

from read_in_files import read_in_files
from NumDaysWeeksMonths import NumDaysWeeksMonths
from Volatility import *
from linear_regression import *
import matplotlib.pyplot as plt
from PastAsPresent import *

from garch_pq_model import GarchModel as gm
# from arch_q_model import ArchModelQ as am
import numpy as np

from function_runs import *

filenames = ['AUDUSD.csv', 'CADUSD.csv', 'CHFUSD.csv', 'EURUSD.csv', 'GBPUSD.csv', 'JPYUSD.csv', 'NOKUSD.csv', 'NZDUSD.csv', 'SEKUSD.csv']
# TODO output tables after each for loop, or store them somehow
for name in filenames:
    # TODO: scale factor for volatility--PLEASE CHECK IF COMPLETED CORRECTLY

    #  reads in the files and puts them into dataframes, returns a dataframe called df
    df, df_single_day, df_single_week, df_single_month = read_in_files(name)
    # THE LINE BELOW IS NOT REALLY NEEDED IF SCALING IS CONSTANT
    # days_weeks_months, num_days_per_year, num_weeks_per_year, num_months_per_year = NumDaysWeeksMonths(df=df)

    daily_vol_result, daily_ret = time_vol_calc(df_single_day)
    weekly_vol_result, weekly_ret = time_vol_calc(df_single_week)
    monthly_vol_result, monthly_ret = time_vol_calc(df_single_month)

    plt.figure(1000)
    plt.plot(daily_vol_result.Date, np.log(daily_vol_result.Volatility_Time))
    # plt.show()
    initial = 0.5

    fc = FunctionCalls()
    Daily = fc.function_runs('Daily', daily_vol_result, 1, [1, 3, 5, 10],
                             [np.array(daily_ret['Return_Time']), 1, 0, int(initial*len(daily_ret))],
                             [np.array(daily_ret['Return_Time']), 1, 1, 0,int(initial*len(daily_ret))])

    Weekly = fc.function_runs('Weekly', weekly_vol_result, 1, [1, 3, 5, 10],
                              [np.array(weekly_ret['Return_Time']), 1, 0, int(initial*len(weekly_ret))],
                               [np.array(weekly_ret['Return_Time']), 1, 1, 0, int(initial*len(weekly_ret))])

    Monthly = fc.function_runs('Monthly', monthly_vol_result, 1, [1, 3, 5, 10],
                               [np.array(monthly_ret['Return_Time']), 1, 0, int(initial*len(monthly_ret))],
                               [np.array(monthly_ret['Return_Time']), 1, 1, 0, int(initial*len(monthly_ret))])


    print("yo")
plt.show()

# TODO: add string to each function, such as "GARCH", or "Daily", or "Weekly" for a more generalized plot
#
# """Past as Present"""
# DAILY_PastAsPresent = PastAsPresent.tn_pred_tn_plus_1(daily_vol_result)
# print("Daily PastAsPresent MSE and QL are: " + str(DAILY_PastAsPresent[0:2]))
#
# WEEKLY_PastAsPresent = PastAsPresent.tn_pred_tn_plus_1(weekly_vol_result)
# print("Weekly PastAsPresent MSE and QL are: " + str(WEEKLY_PastAsPresent[0:2]))
# print("Weekly PastAsPresent MSE and QL are: " + str(WEEKLY_PastAsPresent[0:2]))
#
# MONTHLY_PastAsPresent = PastAsPresent.tn_pred_tn_plus_1(monthly_vol_result)
# print("Monthly PastAsPresent MSE and QL are: " + str(MONTHLY_PastAsPresent[0:2]))
#
#
# """Linear Regression"""
# """DAILY"""
# DAILY_one_lag_results = LinRegression.lin_reg(daily_vol_result, 1)
# print("Daily 1 Lag's MSE and QL are: " + str(DAILY_one_lag_results[0:2]))
#
# DAILY_three_lag_results = LinRegression.lin_reg(daily_vol_result, 3)
# print("Daily 3 Lag's MSE and QL are: " + str(DAILY_three_lag_results[0:2]))
#
# DAILY_five_lag_results = LinRegression.lin_reg(daily_vol_result, 5)
# print("Daily 5 Lag's MSE and QL are: " + str(DAILY_five_lag_results[0:2]))
#
# DAILY_ten_lag_results = LinRegression.lin_reg(daily_vol_result, 10)
# print("Daily 10 Lag's MSE and QL are: " + str(DAILY_ten_lag_results[0:2]))
#
# """WEEK"""
# WEEKLY_one_lag_results = LinRegression.lin_reg(weekly_vol_result, 1)
# print("WEEKLY 1 Lag's MSE and QL are: " + str(WEEKLY_one_lag_results[0:2]))
#
# WEEKLY_three_lag_results = LinRegression.lin_reg(weekly_vol_result, 3)
# print("WEEKLY 3 Lag's MSE and QL are: " + str(WEEKLY_three_lag_results[0:2]))
#
# WEEKLY_five_lag_results = LinRegression.lin_reg(weekly_vol_result, 5)
# print("WEEKLY 5 Lag's MSE and QL are: " + str(WEEKLY_five_lag_results[0:2]))
#
# WEEKLY_ten_lag_results = LinRegression.lin_reg(weekly_vol_result, 10)
# print("WEEKLY 10 Lag's MSE and QL are: " + str(WEEKLY_ten_lag_results[0:2]))
#
# """MONTH"""
# MONTHLY_one_lag_results = LinRegression.lin_reg(monthly_vol_result, 1)
# print("MONTHLY 1 Lag's MSE and QL are: " + str(MONTHLY_one_lag_results[0:2]))
#
# MONTHLY_three_lag_results = LinRegression.lin_reg(monthly_vol_result, 3)
# print("MONTHLY 3 Lag's MSE and QL are: " + str(MONTHLY_three_lag_results[0:2]))
#
# MONTHLY_five_lag_results = LinRegression.lin_reg(monthly_vol_result, 5)
# print("MONTHLY 5 Lag's MSE and QL are: " + str(MONTHLY_five_lag_results[0:2]))
#
# MONTHLY_ten_lag_results = LinRegression.lin_reg(monthly_vol_result, 10)
# print("MONTHLY 10 Lag's MSE and QL are: " + str(MONTHLY_ten_lag_results[0:2]))

# #  TODO FIX THE SCALING FOR ARCH AND GARCH
# """ARCH"""
# initialsize=3
#
DAILY_arch1_mse = gm.arch_q_mse(daily_vol_result, np.array(daily_ret['Return_Time']), 1, 0,initialsize)
# print("Daily ARCH(1) MSE and QL are:" + str(DAILY_arch1_mse))
#
# WEEKLY_arch1_mse = gm.arch_q_mse(weekly_vol_result, np.array(weekly_ret['Return_Time']), 1, 0,initialsize)
# print("WEEKLY ARCH(1) MSE and QL are:" + str(WEEKLY_arch1_mse))
#
# MONTHLY_arch1_mse = gm.arch_q_mse(monthly_vol_result, np.array(monthly_ret['Return_Time']), 1, 0,initialsize)
# print("MONTHLY ARCH(1) MSE and QL are:" + str(MONTHLY_arch1_mse))
#
#
# """GARCH(p,q)"""
# DAILY_garch11_mse = gm.garch_pq_mse(daily_vol_result, np.array(daily_ret['Return_Time']), 1, 1, 0,initialsize)
# print("Daily GARCH(1,1) MSE and QL are:" + str(DAILY_garch11_mse))
#
# WEEKLY_garch11_mse = gm.garch_pq_mse(weekly_vol_result, np.array(weekly_ret['Return_Time']), 1, 1, 0,initialsize)
# print("WEEKLY GARCH(1,1) MSE and QL are:" + str(WEEKLY_garch11_mse))
#
# MONTHLY_garch11_mse = gm.garch_pq_mse(monthly_vol_result, np.array(monthly_ret['Return_Time']), 1, 1, 0,initialsize)
# print("MONTHLY GARCH(1,1) MSE and QL are:" + str(MONTHLY_garch11_mse))
#
#
# # show plots at the very end
# plt.show()
#
# print("hi")
