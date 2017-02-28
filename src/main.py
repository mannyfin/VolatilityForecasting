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
import matplotlib.backends.backend_pdf

# filenames = ['AUDUSD.csv', 'CADUSD.csv', 'CHFUSD.csv', 'EURUSD.csv', 'GBPUSD.csv', 'JPYUSD.csv', 'NOKUSD.csv', 'NZDUSD.csv', 'SEKUSD.csv']
filenames = ['CADUSD.csv']
# TODO output tables after each for loop, or store them somehow
for name in filenames:
    # TODO: scale factor for volatility--PLEASE CHECK IF COMPLETED CORRECTLY

    #  reads in the files and puts them into dataframes, returns a dataframe called df
    df, df_single_day, df_single_week, df_single_month = read_in_files(name)
    # THE LINE BELOW IS NOT REALLY NEEDED IF SCALING IS CONSTANT
    # days_weeks_months, num_days_per_year, num_weeks_per_year, num_months_per_year = NumDaysWeeksMonths(df=df)

    # We use this line below for the name of the graph
    name = name.split('.')[0]

    daily_vol_result, daily_ret = time_vol_calc(df_single_day)
    weekly_vol_result, weekly_ret = time_vol_calc(df_single_week)
    monthly_vol_result, monthly_ret = time_vol_calc(df_single_month)

    plt.figure(1000)
    plt.plot(daily_vol_result.Date, np.log(daily_vol_result.Volatility_Time))
    # plt.show()
    initial = 0.5 # set the first 50% of the input data as in-sample data to fit the model

    fc = FunctionCalls()
    Daily = fc.function_runs(name, 'Daily', daily_vol_result, 1, [1, 3, 5, 10],
                             [np.array(daily_ret['Return_Time']), 1, 0, int(initial*len(daily_ret))],
                             [np.array(daily_ret['Return_Time']), 1, 1, 0,int(initial*len(daily_ret))])

    Weekly = fc.function_runs(name, 'Weekly', weekly_vol_result, 1, [1, 3, 5, 10],
                              [np.array(weekly_ret['Return_Time']), 1, 0, int(initial*len(weekly_ret))],
                               [np.array(weekly_ret['Return_Time']), 1, 1, 0, int(initial*len(weekly_ret))])

    Monthly = fc.function_runs(name, 'Monthly', monthly_vol_result, 1, [1, 3, 5, 10],
                               [np.array(monthly_ret['Return_Time']), 1, 0, int(initial*len(monthly_ret))],
                               [np.array(monthly_ret['Return_Time']), 1, 1, 0, int(initial*len(monthly_ret))])


    print("yo")
plt.show()
# """Output multiple plots into a pdf file"""
# pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")
# for fig in range(1, 22): ## will open an empty extra figure :(
#     pdf.savefig( fig )
# pdf.close()

# TODO: add string to each function, such as "GARCH", or "Daily", or "Weekly" for a more generalized plot
