
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
# from KNNmethods import KNN
from function_runs import *

from VAR2 import *


import matplotlib.backends.backend_pdf
print("hi")
filenamesx = ['AUDUSD.csv', 'CADUSD.csv',  'CHFUSD.csv', 'EURUSD.csv', 'GBPUSD.csv', 'JPYUSD.csv', 'NOKUSD.csv', 'NZDUSD.csv', 'SEKUSD.csv']
#filenamesx = ['SEKUSD.csv','CADUSD.csv','CHFUSD.csv']
#filenames = ['NZDUSD.csv']#,'CADUSD.csv']

KNN_test = []

for filenames in filenamesx:
    dailyvol_zeroes= pd.DataFrame()
    weeklyvol_zeroes= pd.DataFrame()
    monthlyvol_zeroes= pd.DataFrame()
    dailyret_zeroes= pd.DataFrame()
    weeklyret_zeroes= pd.DataFrame()
    monthlyret_zeroes = pd.DataFrame()
    Daily_list = list()
    namelist = list()
    for count, name in enumerate([filenames]):
        #  reads in the files and puts them into dataframes, returns a dataframe called df
        df, df_single_day, df_single_week, df_single_month = read_in_files(name)
        # THE LINE BELOW IS NOT REALLY NEEDED IF SCALING IS CONSTANT
        # days_weeks_months, num_days_per_year, num_weeks_per_year, num_months_per_year = NumDaysWeeksMonths(df=df)
        # We use this line below for the name of the graph
        name = name.split('.')[0]
        namelist.append(name)
        print("Running file: " + str(name))
        warmup_period = 100
        daily_vol_result, daily_ret, daily_vol_zeroes, daily_ret_zeroes = time_vol_calc(df_single_day)
        # weekly_vol_result, weekly_ret, weekly_vol_zeroes, weekly_ret_zeroes = time_vol_calc(df_single_week)
        # monthly_vol_result, monthly_ret, monthly_vol_zeroes, monthly_ret_zeroes = time_vol_calc(df_single_month)

        # be careful of the underscore, _
        dailyvol_zeroes = pd.concat([dailyvol_zeroes, daily_vol_zeroes], axis=1)

        # dailyvol_zeroes = pd.concat([dailyvol_zeroes, daily_vol_zeroes['Volatility_Time']], axis=1)

        dailyvol_zeroes.rename(columns={'Volatility_Time': name}, inplace=True)

        # be careful of the underscore, _
        dailyret_zeroes = pd.concat([dailyret_zeroes, daily_ret_zeroes], axis=1)

        dailyret_zeroes.rename(columns={'Return_Time': name}, inplace=True)

        warmup_period_daily = 400
        warmup_period_weekly = 70
        warmup_period_monthly = 24


    print("hi")
    # does not have zeroes
    daily_vol_combined = dailyvol_zeroes[(dailyvol_zeroes != 0).all(1)]

    try:
        dates = daily_vol_combined.Date.loc[:, ~daily_vol_combined.Date.columns.duplicated()].reset_index()
        daily_vol_combined.drop('Date', axis=1, inplace=True)
        daily_vol_combined=daily_vol_combined.apply(pd.to_numeric)

    except:
        dates = daily_vol_combined.Date
    # drop duplicate columns

    # weekly_vol_combined = weeklyvol_zeroes[(weeklyvol_zeroes != 0).all(1)]
    # monthly_vol_combined = monthlyvol_zeroes[(monthlyvol_zeroes != 0).all(1)]
    """I am referencing the $Time$vol_zeroes variable in the lines below because there are (or could be) days where the ret
      is zero"""
    daily_ret_combined = dailyret_zeroes[(dailyvol_zeroes != 0).all(1)]
    # weekly_ret_combined = weeklyret_zeroes[(weeklyvol_zeroes != 0).all(1)]
    # monthly_ret_combined = monthlyret_zeroes[(monthlyvol_zeroes != 0).all(1)]

    # We need to reset index
    daily_vol_combined.reset_index(drop=True, inplace=True)
    # weekly_vol_combined.reset_index(drop=True, inplace=True)
    # monthly_vol_combined.reset_index(drop=True, inplace=True)
    daily_ret_combined.reset_index(drop=True, inplace=True)

    p=3
    # use this below
    fc = FunctionCalls()
    # xmat = pd.DataFrame([sum([daily_vol_combined[currency].loc[i+p-1:i:-1].as_matrix().tolist() for currency in daily_vol_combined.keys()],[]) for i in range(len(daily_vol_combined)-p)])

    testwhat = "KNN"

    training_sample, test_sample = daily_vol_combined.iloc[0:910].copy(), daily_vol_combined.iloc[910:].copy()

    training_sample.reset_index(drop=True)
    training_date = training_sample["Date"]

    test_sample.reset_index(drop=True)
    test_date = test_sample['Date']

    training = True
    # do not use the line below to test VAR, use main.py, not mainMethods.py
    if testwhat == "VAR":
        VAR_test = fc.function_runs(dates=dates, filename='Combined Curr.', stringinput='Daily', warmup=100, input_data=np.log(daily_vol_combined), var_q=[1, 2, 3])
    else:
        if training:
            KNN_training = [ [ fc.function_runs(dates=training_date, filename=str(name)+' Single Knn.', stringinput='Daily',warmup=warmup,input_data=test_sample,k_nn=[i])
            for i in np.arange(2,21) ] for warmup in np.arange(100,200,500)]
        else:
            KNN_test.append(fc.function_runs(dates=test_date, filename=filenames, stringinput='Daily',warmup=100,input_data=test_sample,k_nn=[20]))
plt.show()
plt.close()

if len(KNN_training) != 1:
    ks = np.arange(2,10)
    warmups = np.arange(50,200,50)


    k,w = np.meshgrid(ks,warmups)
    SEkw = np.array([ [ KNN_training[warmup][i-2]['MSE'][0] for i in np.arange(2,10) ] for warmup in np.arange(len(np.arange(50,200,50)))])
    k,w = np.meshgrid(ks,warmups)

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    matplotlib.style.use('ggplot')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    sur=ax.plot_surface(k,w,SEkw,cmap=cm.coolwarm,linewidth=0,antialiased=False,rstride=1,cstride=1)
    fig.colorbar(sur,shrink=0.5,aspect=5)

    ax.set_xlabel('k')
    ax.set_ylabel('warmup')
    ax.set_zlabel('SE')
    plt.title(str(filenames[0]))
    plt.show()


# """Output multiple plots into a pdf file"""
# pdf = matplotlib.backends.backend_pdf.PdfPages(name+".pdf")
# for fig in range(1, 3*count+3+1):
#     pdf.savefig( fig, dpi=1200 )
# pdf.close()
#
#


print("Complete")
