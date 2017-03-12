# This main file is a sketch of the overall program
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
import matplotlib.backends.backend_pdf

filenames = ['AUDUSD.csv', 'CADUSD.csv',  'CHFUSD.csv', 'EURUSD.csv', 'GBPUSD.csv', 'JPYUSD.csv', 'NOKUSD.csv', 'NZDUSD.csv', 'SEKUSD.csv']
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
    # THE LINE BELOW IS NOT REALLY NEEDED IF SCALING IS CONSTANT
    # days_weeks_months, num_days_per_year, num_weeks_per_year, num_months_per_year = NumDaysWeeksMonths(df=df)
    # We use this line below for the name of the graph
    name = name.split('.')[0]
    namelist.append(name)
    print("Running file: " + str(name))
    warmup_period = 400
    daily_vol_result, daily_ret, daily_vol_zeroes, daily_ret_zeroes = time_vol_calc(df_single_day)
    # weekly_vol_result, weekly_ret, weekly_vol_zeroes, weekly_ret_zeroes = time_vol_calc(df_single_week)
    # monthly_vol_result, monthly_ret, monthly_vol_zeroes, monthly_ret_zeroes = time_vol_calc(df_single_month)

    # be careful of the underscore, _
    # dailyvol_zeroes = pd.concat([dailyvol_zeroes, daily_vol_zeroes['Volatility_Time']], axis=1)
    # dailyvol_zeroes.rename(columns={'Volatility_Time': name}, inplace=True)
    # weeklyvol_zeroes = pd.concat([weeklyvol_zeroes, weekly_vol_zeroes['Volatility_Time']], axis=1)
    # weeklyvol_zeroes.rename(columns={'Volatility_Time': name}, inplace=True)
    # monthlyvol_zeroes = pd.concat([monthlyvol_zeroes, monthly_vol_zeroes['Volatility_Time']], axis=1)
    # monthlyvol_zeroes.rename(columns={'Volatility_Time': name}, inplace=True)
    # # be careful of the underscore, _
    # dailyret_zeroes = pd.concat([dailyret_zeroes, daily_ret_zeroes['Return_Time']], axis=1)
    # dailyret_zeroes.rename(columns={'Return_Time': name}, inplace=True)
    # weeklyret_zeroes = pd.concat([weeklyret_zeroes, weekly_ret_zeroes['Return_Time']], axis=1)
    # weeklyret_zeroes.rename(columns={'Return_Time': name}, inplace=True)
    # monthlyret_zeroes = pd.concat([monthlyret_zeroes, monthly_ret_zeroes['Return_Time']], axis=1)
    # monthlyret_zeroes.rename(columns={'Return_Time': name}, inplace=True)
    # """
    # testing KNN
    # """

    # plt.figure(1, figsize=(12, 7))
    # fc = FunctionCalls()
    # Daily = fc.function_runs(filename=name, stringinput='Daily', warmup=warmup_period, input_data=daily_vol_result[1:], k_nn=50)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True, ncol=3)


#     plt.figure(len(filenames)*21+1+count)
#     plt.plot(daily_vol_result.Date, np.log(daily_vol_result.Volatility_Time))
#     plt.title('Daily Vol Result for ' + str(name))
#     plt.ylabel('Ln(Volatility)')
#     # plt.show()
#
    warmup_period = 400
    plt.figure(3*count+1, figsize=(12, 7))
    fc = FunctionCalls()
    Daily = fc.function_runs(filename=name, stringinput='Daily', warmup=warmup_period, input_data=daily_vol_result[1:],
                             tnplus1=1, lr=[1, 3, 5, 10], arch=[np.array(daily_ret['Return_Time'][1:]), 1, 0],
                             garchpq=[np.array(daily_ret['Return_Time'][1:]), 1, 1, 0], k_nn=10)

    Daily_list.append(Daily)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True, ncol=3)
#
#     plt.figure(3*count+2, figsize=(12, 7))
#     Weekly = fc.function_runs(filename=name, stringinput='Weekly', warmup=warmup_period,
#                               input_data=weekly_vol_result[1:-2], tnplus1=1, lr=[1, 3, 5, 10],
#                               arch=[np.array(weekly_ret['Return_Time'][1:-2]), 1, 0],
#                               garchpq=[np.array(weekly_ret['Return_Time'][1:-2]), 1, 1, 0])
#
#     plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True, ncol=3)
#
#     plt.figure(3*count+3, figsize=(12, 7))
#     Monthly = fc.function_runs(filename=name, stringinput='Monthly', warmup=warmup_period, input_data=monthly_vol_result[1:],
#                                tnplus1=1, lr=[1, 3, 5, 10], arch=[np.array(monthly_ret['Return_Time'][1:]), 1, 0],
#                                garchpq=[np.array(monthly_ret['Return_Time'][1:]), 1, 1, 0])
#
    # tablegen(Daily)
    # tablegen(Weekly)
    # tablegen(Monthly)
#     plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True, ncol=3)
#     # plt.hold(False)
Daily_df = pd.DataFrame()
Daily_df = pd.concat(Daily_list, axis=1, keys=namelist)

# to make index slices:::
"""
These will slice all the MSE and QL columns respectively
"""
idx = pd.IndexSlice
Daily_df.loc[idx[:], idx[:, 'MSE']].plot(figsize=[13, 7], logy=True)
plt.legend()   #  I add this line becuase the legend shows (None,None) as the first entry. Regenerating the legend fixes this

Daily_df.loc[idx[:], idx[:, 'QL']].plot(figsize=[13, 7], logy=True)
plt.legend()

"This sums up all the rows. And so it is the sum of the MSE or QL for all currencies for a particular model"
MSE_sumdaily= pd.DataFrame(Daily_df.loc[idx[:], idx[:, 'MSE']].sum(axis=1), columns=['Sum of MSE [Daily]']).plot(figsize=[13, 7], table=True)
MSE_sumdaily.get_xaxis().set_visible(False)
QL_sumdaily= pd.DataFrame(Daily_df.loc[idx[:], idx[:, 'QL']].sum(axis=1), columns=['Sum of QL [Daily]']).plot(figsize=[13, 7], table=True)
QL_sumdaily.get_xaxis().set_visible(False)


print("hi")
# does not have zeroes
daily_vol_combined = dailyvol_zeroes[(dailyvol_zeroes != 0).all(1)]
weekly_vol_combined = weeklyvol_zeroes[(weeklyvol_zeroes != 0).all(1)]
monthly_vol_combined = monthlyvol_zeroes[(monthlyvol_zeroes != 0).all(1)]
"""I am referencing the $Time$vol_zeroes variable in the lines below because there are (or could be) days where the ret
  is zero"""
daily_ret_combined = dailyret_zeroes[(dailyvol_zeroes != 0).all(1)]
weekly_ret_combined = weeklyret_zeroes[(weeklyvol_zeroes != 0).all(1)]
monthly_ret_combined = monthlyret_zeroes[(monthlyvol_zeroes != 0).all(1)]

# We need to reset index
daily_vol_combined.reset_index(drop=True, inplace=True)
weekly_vol_combined.reset_index(drop=True, inplace=True)
monthly_vol_combined.reset_index(drop=True, inplace=True)
daily_ret_combined.reset_index(drop=True, inplace=True)
weekly_ret_combined.reset_index(drop=True, inplace=True)
monthly_ret_combined.reset_index(drop=True, inplace=True)


# """Output multiple plots into a pdf file"""
# pdf = matplotlib.backends.backend_pdf.PdfPages(name+".pdf")
# for fig in range(1, 3*count+3+1):
#     pdf.savefig( fig, dpi=1200 )
# pdf.close()
#
#
print("hi")
print("Complete")