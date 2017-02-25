# This main file is a sketch of the overall program
#  if you dont have this installed then here's how to install the module:
#  conda install -c https://conda.binstar.org/bashtage arch

from read_in_files import read_in_files
from NumDaysWeeksMonths import NumDaysWeeksMonths
from Volatility import *
from linear_regression import *
import matplotlib.pyplot as plt
from PastAsPresent import *
from RetCalculator import *
from sklearn.metrics import mean_squared_error as mse
from garch_pq_model import garch_model as gm
from arch_q_model import arch_model as am




filenames = 'AUDUSD.csv'
#  reads in the files and puts them into dataframes, returns a dataframe called df
df, df_single_day, df_single_month = read_in_files(filenames)
days_weeks_months, num_days_per_year = NumDaysWeeksMonths(df=df)
daily_vol_result = daily_vol_calc(df, df_single_day, num_days_per_year)
daily_ret = RetCalculator.daily_ret_df(df, df_single_day, num_days_per_year)
#TODO: add monthly_vol_result
#TODO: add weekly_vol_result


MSE_PastAsPresent = PastAsPresent.today_tomorrow(daily_vol_result)
print("Daily PastAsPresent MSE is " + str(MSE_PastAsPresent))


one_lag_results = LinRegression.lin_reg(daily_vol_result, 1)
print("Daily 1 Lag's MSE is " + str(one_lag_results[0]))

three_lag_results = LinRegression.lin_reg(daily_vol_result, 3)
print("Daily 3 Lag's MSE is " + str(three_lag_results[0]))

five_lag_results = LinRegression.lin_reg(daily_vol_result, 5)
print("Daily 5 Lag's MSE is " + str(one_lag_results[0]))

ten_lag_results = LinRegression.lin_reg(daily_vol_result, 10)
print("Daily 10 Lag's MSE is " + str(ten_lag_results[0]))

daily_arch1_mse = am.arch_q_mse(daily_vol_result,np.array(daily_ret['Return_Daily']),1,1)
print("Daily ARCH(1,1) MSE is :" + str(daily_arch1_mse))


daily_garch11_mse = gm.garch_pq_mse(daily_vol_result,np.array(daily_ret['Return_Daily']),1,1,1)
print("Daily GARCH(1,1) MSE is :" + str(daily_garch11_mse))


print("hi")



# DailyVolDF(df_single_day, num_days_per_year)


# predict tomorrows volatility based on today (same for weekly and monthly
# #     -this also looks like linear regression
#
# # perform linear regression (using past 1, 3, 5, vols)
# linear_regression()
#
# # perform ARCH
# arch_model()
#
# # perform GARCH(1,1)
# garch11_model()
#
# errorplots()
#

