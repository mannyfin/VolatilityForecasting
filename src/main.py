
# This main file is a sketch of the overall program
#  if you dont have this installed then here's how to install the module:
#  conda install -c https://conda.binstar.org/bashtage arch

from read_in_files import read_in_files
from NumDaysWeeksMonths import NumDaysWeeksMonths
from Volatility import *
from linear_regression import *
import matplotlib.pyplot as plt
# import linear_regression
# from part1 import *

filenames = 'AUDUSD.csv'
#  reads in the files and puts them into dataframes, returns a dataframe called df
df, df_single_day = read_in_files(filenames)
days_weeks_months, num_days_per_year = NumDaysWeeksMonths(df=df)
daily_vol_result = daily_vol_calc(df, df_single_day, num_days_per_year)

# part1.today_tomorrow(daily_vol_result)

one_day_results = LinRegression.one_day_trailing(daily_vol_result)

three_day_results = LinRegression.three_day_trailing(daily_vol_result)

five_day_results = LinRegression.five_day_trailing(daily_vol_result)
plt.show()
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
