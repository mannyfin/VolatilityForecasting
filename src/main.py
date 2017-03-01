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
import warnings
warnings.filterwarnings("ignore")
from function_runs import *
import matplotlib.backends.backend_pdf

# filenames = ['AUDUSD.csv', 'CADUSD.csv', 'CHFUSD.csv', 'EURUSD.csv', 'GBPUSD.csv', 'JPYUSD.csv', 'NOKUSD.csv', 'NZDUSD.csv', 'SEKUSD.csv']
filenames = ['CHFUSD.csv']
# TODO output tables after each for loop, or store them somehow
for count, name in enumerate(filenames):
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

    # plt.figure(len(filenames)*21+1+count)
    # plt.plot(daily_vol_result.Date, np.log(daily_vol_result.Volatility_Time))
    # plt.title('Daily Vol Result for ' + str(name))
    # plt.ylabel('Ln(Volatility)')
    # plt.show()
    warmup_period = 10 # set the first 50% of the input data as in-sample data to fit the model
    plt.figure(1, figsize=(12, 5))
    fc = FunctionCalls()
    Daily = fc.function_runs(filename=name, stringinput='Daily', warmup=warmup_period, input_data=daily_vol_result[1:],
                             tnplus1=1, lr=[1, 3, 5, 10], arch=[np.array(daily_ret['Return_Time'][1:]), 1, 0],
                             garchpq=[np.array(daily_ret['Return_Time'][1:]), 1, 1, 0])

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fancybox=True, shadow=True, ncol=3)
    # plt.hold(False)
    plt.figure(2, figsize=(12, 5))
    Weekly = fc.function_runs(filename=name, stringinput='Weekly', warmup=warmup_period,
                              input_data=weekly_vol_result[1:-2], tnplus1=1, lr=[1, 3, 5, 10],
                              arch=[np.array(weekly_ret['Return_Time'][1:-2]), 1, 0],
                              garchpq=[np.array(weekly_ret['Return_Time'][1:-2]), 1, 1, 0])

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fancybox=True, shadow=True, ncol=3)
    # plt.hold(False)
    plt.figure(3, figsize=(12, 5))
    Monthly = fc.function_runs(filename=name, stringinput='Monthly', warmup=warmup_period, input_data=monthly_vol_result[1:],
                               tnplus1=1, lr=[1, 3, 5, 10], arch=[np.array(monthly_ret['Return_Time'][1:]), 1, 0],
                               garchpq=[np.array(monthly_ret['Return_Time'][1:]), 1, 1, 0])

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fancybox=True, shadow=True, ncol=3)
    # plt.hold(False)
    print("yo")
# plt.show()
print("yo")
"""Output multiple plots into a pdf file"""
pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")
# for fig in range(1, len(filenames)*21+1+count): ## will open an empty extra figure :(
for fig in range(1, 4): ## will open an empty extra figure :(
    pdf.savefig( fig, dpi=1200 )
pdf.close()

plt.show()

print("yo")