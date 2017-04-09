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
from function_runs import *
import os
from LogisticReg_SVM_KernelSVM import *

from VAR_new import *
from returnvoldf import retvoldf
from preprocess import preprocess_data
# please install python-pptx with pip install python-pptx
from PPT import *




print("hi")
filenames = ['AUDUSD.csv', 'CADUSD.csv',  'CHFUSD.csv', 'EURUSD.csv', 'GBPUSD.csv', 'JPYUSD.csv', 'NOKUSD.csv', 'NZDUSD.csv', 'SEKUSD.csv']

os.chdir('Data')
v = pd.read_csv('v.csv')
v.columns = ['Date', 'value']
v = v.set_index('Date')
os.chdir('..')
dailyvol_zeroes = pd.DataFrame()
weeklyvol_zeroes = pd.DataFrame()
monthlyvol_zeroes = pd.DataFrame()
dailyret_zeroes = pd.DataFrame()
weeklyret_zeroes = pd.DataFrame()
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
    weekly_vol_result, weekly_ret, weekly_vol_zeroes, weekly_ret_zeroes = time_vol_calc(df_single_week)
    # monthly_vol_result, monthly_ret, monthly_vol_zeroes, monthly_ret_zeroes = time_vol_calc(df_single_month)

    # be careful of the underscore, _
    dailyvol_zeroes = pd.concat([dailyvol_zeroes, daily_vol_zeroes], axis=1)

    # dailyvol_zeroes = pd.concat([dailyvol_zeroes, daily_vol_zeroes['Volatility_Time']], axis=1)

    dailyvol_zeroes.rename(columns={'Volatility_Time': name}, inplace=True)

    weeklyvol_zeroes = pd.concat([weeklyvol_zeroes, weekly_vol_zeroes['Volatility_Time']], axis=1)
    weeklyvol_zeroes.rename(columns={'Volatility_Time': name}, inplace=True)
    # monthlyvol_zeroes = pd.concat([monthlyvol_zeroes, monthly_vol_zeroes['Volatility_Time']], axis=1)
    # monthlyvol_zeroes.rename(columns={'Volatility_Time': name}, inplace=True)

    # be careful of the underscore, _
    dailyret_zeroes = pd.concat([dailyret_zeroes, daily_ret_zeroes], axis=1)
    weeklyret_zeroes = pd.concat([weeklyret_zeroes, weekly_ret_zeroes], axis=1)

    # dailyret_zeroes = pd.concat([dailyret_zeroes, daily_ret_zeroes['Return_Time']], axis=1)
    dailyret_zeroes.rename(columns={'Return_Time': name}, inplace=True)
    weeklyret_zeroes.rename(columns={'Return_Time': name}, inplace=True)

    "returnvoldf"
    "We want to test daily and weekly data"

    # preprocess_daily = retvoldf(daily_ret, daily_vol_result, v)
    preprocess_daily, test_sample_daily, train_sample_daily = retvoldf(daily_ret, daily_vol_result, v)
    preprocess_weekly,test_sample_weekly, train_sample_weekly = retvoldf(weekly_ret, weekly_vol_result, v)

    # model can take inputs "LogisticRegression", "SVM", "KernelSVM_poly" ,"KernelSVM_rbf" or "KernelSVM_sigmoid"
    DeltaSeq = np.exp(np.linspace(-10, -2, num=20))
    # DeltaSeq = np.exp(np.linspace(-10, -2, num=20))
    p_seq = np.array([3, 5, 10])
    q_seq = np.array([3, 5, 10])
    """
    for forecaster = 1, no p and q, try different deg for KernalSVM poly     # try deg = 2,3, 4, 5
    for forecaster = 2, no p and q, try different deg for KernalSVM poly     # try deg = 2,3, 4, 5
    for forecaster = 3, p=3,5,10 and NO q, try different deg for KernalSVM poly     # try deg = 2,3, 4, 5.
    for forecaster = 4, p=3,5,10 and q=3,5,10, try different deg for KernalSVM poly     # try deg = 2,3, 4, 5.

    For weekly data, reduce warmup in MSE_QL_SE_Test from 400 to another, and warmup_test from 100 to another value
    passed daily or weekly into file, warmup weekly = 80, 30 for test set
    """
    if not os.path.exists(name):
        os.mkdir(name)

    warmup_period_for_daily = 100  # size of the rolling window for daily data
    warmup_period_for_weekly = 50  # size of the rolling window for weekly data

    os.chdir(name)

    ModelTypes = ["LogisticRegression", "SVM", "KernelSVM_poly", "KernelSVM_poly",
                  "KernelSVM_poly", "KernelSVM_poly", "KernelSVM_rbf", "KernelSVM_sigmoid"]
    deg = [None, None, 2, 3, 4, 5, None, None]

    MSE_Test_Outputs_daily = []
    QL_Test_Outputs_daily = []
    MSE_Test_Outputs_weekly = []
    QL_Test_Outputs_weekly = []
    # for k in range(1, 5):
    for k in range(1,6):
        for i in range(len(ModelTypes)):
            if k < 3 or k==5:
                input_p_seq = None
            # elif k >= 3 or k!=5:
            elif k==3 or k==4:
                input_p_seq = p_seq

            if k < 4 or k==5:
                input_q_seq = None
            elif k ==4:
                input_q_seq = q_seq
            #
            Results_daily = MSE_QL_SE_Test(preprocess, DeltaSeq, warmup_test=warmup_period_for_daily, filename=name,
                                           model=ModelTypes[i], deg=deg[i], forecaster=k, p_seq=input_p_seq,
                                           q_seq=input_q_seq,stringinput='Daily')
            Results_weekly = MSE_QL_SE_Test(preprocess_w, DeltaSeq, warmup_test=warmup_period_for_weekly, filename=name,
                                            model=ModelTypes[i], deg=deg[i], forecaster=k, p_seq=input_p_seq,
                                            q_seq=input_q_seq,stringinput='Weekly')

            MSE_Test_Outputs_daily.append(Results_daily[0])
            QL_Test_Outputs_daily.append(Results_daily[1])
            MSE_Test_Outputs_weekly.append(Results_weekly[0])
            QL_Test_Outputs_weekly.append(Results_weekly[1])

    # making a table
    ModelName1 = name+"_Logit_forecaster1"
    ModelName2 = name+"_SVM_forecaster1"
    ModelName3 = name+"_KernelSVM_poly_deg_2_forecaster1"
    ModelName4 = name+"_KernelSVM_poly_deg_3_forecaster1"
    ModelName5 = name+"_KernelSVM_poly_deg_4_forecaster1"
    ModelName6 = name+"_KernelSVM_poly_deg_5_forecaster1"
    ModelName7 = name+"_KernelSVM_rbf_forecaster1"
    ModelName8 = name+"_KernelSVM_sigmoid_forecaster1"

    ModelNames = [ModelName1, ModelName2, ModelName3, ModelName4, ModelName5, ModelName6, ModelName7,ModelName8,
                  ModelName1.replace("1", "2"), ModelName2.replace("1", "2"), ModelName3.replace("1", "2"),
                  ModelName4.replace("1", "2"),
                  ModelName5.replace("1", "2"), ModelName6.replace("1", "2"), ModelName7.replace("1", "2"),
                  ModelName8.replace("1", "2"),
                  ModelName1.replace("1", "3"), ModelName2.replace("1", "3"), ModelName3.replace("1", "3"),
                  ModelName4.replace("1", "3"),
                  ModelName5.replace("1", "3"), ModelName6.replace("1", "3"), ModelName7.replace("1", "3"),
                  ModelName8.replace("1", "3"),
                  ModelName1.replace("1", "4"), ModelName2.replace("1", "4"), ModelName3.replace("1", "4"),
                  ModelName4.replace("1", "4"),
                  ModelName5.replace("1", "4"), ModelName6.replace("1", "4"), ModelName7.replace("1", "4"),
                  ModelName8.replace("1", "4"),
                  ModelName1.replace("1", "5"), ModelName2.replace("1", "5"), ModelName3.replace("1", "5"),
                  ModelName4.replace("1", "5"),
                  ModelName5.replace("1", "5"), ModelName6.replace("1", "5"), ModelName7.replace("1", "5"),
                  ModelName8.replace("1", "5")]

    df_output_collction_daily = {'Model Type': ModelNames,
                                 'Test Sample_MSE_Daily': MSE_Test_Outputs_daily,
                                 'Test Sample_QL_Daily': QL_Test_Outputs_daily}
    df_output_collction_weekly = {'Model Type': ModelNames,
                                  'Test Sample_MSE_Weekly': MSE_Test_Outputs_weekly,
                                  'Test Sample_QL_Weekly': QL_Test_Outputs_weekly}


    df_logistic_SVM_KernelSVM_daily = pd.DataFrame(df_output_collction_daily,
                                                   columns=['Model Type','Test Sample_MSE_Daily', 'Test Sample_QL_Daily'])
    df_logistic_SVM_KernelSVM_weekly = pd.DataFrame(df_output_collction_weekly,
                                                    columns=['Model Type','Test Sample_MSE_Weekly', 'Test Sample_QL_Weekly'])


    df_logistic_SVM_KernelSVM_daily.to_csv(name+'df_logistic_SVM_KernelSVM_daily.csv')
    df_logistic_SVM_KernelSVM_weekly.to_csv(name+'df_logistic_SVM_KernelSVM_weekly.csv')


    for filename in filenames:
        filename_new = filename.replace(".csv", "")
        Output_to_PPT(filename_new)

    os.chdir('..')


print("Complete")