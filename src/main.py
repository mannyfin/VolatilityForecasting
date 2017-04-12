#  if you dont have this installed then here's how to install the module:
#  conda install -c https://conda.binstar.org/bashtage arch

from read_in_files import read_in_files
from Volatility import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# from VAR_new import *
from returnvoldf import retvoldf
# please install python-pptx with pip install python-pptx
from PPT import *
from LogisticRegression import *
from LogisticRegression_MA import *
from SVM import *

print("hi")
filenames = ['AUDUSD.csv', 'CADUSD.csv',  'CHFUSD.csv', 'EURUSD.csv', 'GBPUSD.csv', 'NOKUSD.csv', 'NZDUSD.csv']

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

LR_ln_SE_collect_df_one_file = pd.DataFrame()
SVM_ln_SE_collect_df_one_file = pd.DataFrame()
# C_seq = np.arange(0.1,2,1)
C_seq = np.arange(0.1,5,0.1)
SVM_models = ["SVM", "KernelSVM_rbf", "KernelSVM_sigmoid"]

for count, name in enumerate(filenames):
    #  reads in the files and puts them into dataframes, returns a dataframe called df
    df, df_single_day, df_single_week, df_single_month = read_in_files(name)
    # THE LINE BELOW IS NOT REALLY NEEDED IF SCALING IS CONSTANT
    # days_weeks_months, num_days_per_year, num_weeks_per_year, num_months_per_year = NumDaysWeeksMonths(df=df)
    # We use this line below for the name of the graph
    name = name.split('.')[0]
    namelist.append(name)
    print("Running file: " + str(name))
    daily_vol_result, daily_ret, daily_vol_zeroes, daily_ret_zeroes = time_vol_calc(df_single_day)
    weekly_vol_result, weekly_ret, weekly_vol_zeroes, weekly_ret_zeroes = time_vol_calc(df_single_week)

    # be careful of the underscore, _
    dailyvol_zeroes = pd.concat([dailyvol_zeroes, daily_vol_zeroes], axis=1)
    dailyvol_zeroes.rename(columns={'Volatility_Time': name}, inplace=True)

    weeklyvol_zeroes = pd.concat([weeklyvol_zeroes, weekly_vol_zeroes['Volatility_Time']], axis=1)
    weeklyvol_zeroes.rename(columns={'Volatility_Time': name}, inplace=True)

    # be careful of the underscore, _
    dailyret_zeroes = pd.concat([dailyret_zeroes, daily_ret_zeroes], axis=1)
    weeklyret_zeroes = pd.concat([weeklyret_zeroes, weekly_ret_zeroes], axis=1)

    # dailyret_zeroes = pd.concat([dailyret_zeroes, daily_ret_zeroes['Return_Time']], axis=1)
    dailyret_zeroes.rename(columns={'Return_Time': name}, inplace=True)
    weeklyret_zeroes.rename(columns={'Return_Time': name}, inplace=True)

    "returnvoldf"
    "We want to test daily and weekly data"

    preprocess_daily, test_sample_daily, train_sample_daily = retvoldf(daily_ret, daily_vol_result, v)
    preprocess_weekly,test_sample_weekly, train_sample_weekly = retvoldf(weekly_ret, weekly_vol_result, v)

    for time in ["Daily","Weekly"]:
        if time == "Daily":
            train_sample = train_sample_daily
            test_sample = test_sample_daily
        elif time =="Weekly":
            train_sample = train_sample_weekly
            test_sample = test_sample_weekly

        LogisticRegression_forecaster1 = test_performance_LR(train_sample, test_sample, forecaster=1)
        LogisticRegression_forecaster2 = test_performance_LR(train_sample, test_sample, forecaster=2)
        LogisticRegression_forecaster3 = test_performance_LR_MA(train_sample, test_sample, forecaster=3, numCV=20, time=time, name=name)
        LogisticRegression_forecaster4 = test_performance_LR_MA(train_sample, test_sample, forecaster=4, numCV=20, time=time, name=name)
        LogisticRegression_forecaster5 = test_performance_LR(train_sample, test_sample, forecaster=5)
        LogisticRegression_forecaster6 = test_performance_LR(train_sample, test_sample, forecaster=6)

        SVM_forecaster1 = []
        SVM_forecaster2 = []
        SVM_forecaster3 = []
        SVM_forecaster4 = []
        SVM_forecaster5 = []
        SVM_forecaster6 = []

        for k in range(len(SVM_models)):
            SVM_forecaster1.append(SVM_test_performance(train_sample, test_sample, forecaster=1, numCV=20,
                                                               model=SVM_models[k], C_seq=C_seq, time=time, name=name))
            SVM_forecaster2.append(SVM_test_performance(train_sample, test_sample, forecaster=2, numCV=20,
                                                          model=SVM_models[k], C_seq=C_seq, time=time, name=name))
            SVM_forecaster3.append(SVM_test_performance(train_sample, test_sample, forecaster=3, numCV=20,
                                                          model=SVM_models[k], C_seq=C_seq, time=time, name=name))
            SVM_forecaster4.append(SVM_test_performance(train_sample, test_sample, forecaster=4, numCV=20,
                                                          model=SVM_models[k], C_seq=C_seq, time=time, name=name))
            SVM_forecaster5.append(SVM_test_performance(train_sample, test_sample, forecaster=5, numCV=20,
                                                          model=SVM_models[k], C_seq=C_seq, time=time, name=name))
            SVM_forecaster6.append(SVM_test_performance(train_sample, test_sample, forecaster=6, numCV=20,
                                                          model=SVM_models[k], C_seq=C_seq, time=time, name=name))

        # output MSE and QL in to csv files for SVM and Kernel SVM Models
        SVM_Optimal_p_all_forecaster_one_file = []
        SVM_Optimal_q_all_forecaster_one_file = []
        SVM_Optimal_C_all_forecaster_one_file = []
        SVM_MSE_collect_one_file = []
        SVM_QL_collect_one_file = []

        for k in range(len(SVM_models)):
            SVM_Optimal_p_all_forecaster_one_file.append(SVM_forecaster1[k][4])
            SVM_Optimal_q_all_forecaster_one_file.append(SVM_forecaster1[k][5])
            SVM_Optimal_C_all_forecaster_one_file.append(SVM_forecaster1[k][3])
            SVM_MSE_collect_one_file.append(SVM_forecaster1[k][0])
            SVM_QL_collect_one_file.append(SVM_forecaster1[k][1])
        for k in range(len(SVM_models)):
            SVM_Optimal_p_all_forecaster_one_file.append(SVM_forecaster2[k][4])
            SVM_Optimal_q_all_forecaster_one_file.append(SVM_forecaster2[k][5])
            SVM_Optimal_C_all_forecaster_one_file.append(SVM_forecaster2[k][3])
            SVM_MSE_collect_one_file.append(SVM_forecaster2[k][0])
            SVM_QL_collect_one_file.append(SVM_forecaster2[k][1])
        for k in range(len(SVM_models)):
            SVM_Optimal_p_all_forecaster_one_file.append(SVM_forecaster3[k][4])
            SVM_Optimal_q_all_forecaster_one_file.append(SVM_forecaster3[k][5])
            SVM_Optimal_C_all_forecaster_one_file.append(SVM_forecaster3[k][3])
            SVM_MSE_collect_one_file.append(SVM_forecaster3[k][0])
            SVM_QL_collect_one_file.append(SVM_forecaster3[k][1])
        for k in range(len(SVM_models)):
            SVM_Optimal_p_all_forecaster_one_file.append(SVM_forecaster4[k][4])
            SVM_Optimal_q_all_forecaster_one_file.append(SVM_forecaster4[k][5])
            SVM_Optimal_C_all_forecaster_one_file.append(SVM_forecaster4[k][3])
            SVM_MSE_collect_one_file.append(SVM_forecaster4[k][0])
            SVM_QL_collect_one_file.append(SVM_forecaster4[k][1])
        for k in range(len(SVM_models)):
            SVM_Optimal_p_all_forecaster_one_file.append(SVM_forecaster5[k][4])
            SVM_Optimal_q_all_forecaster_one_file.append(SVM_forecaster5[k][5])
            SVM_Optimal_C_all_forecaster_one_file.append(SVM_forecaster5[k][3])
            SVM_MSE_collect_one_file.append(SVM_forecaster5[k][0])
            SVM_QL_collect_one_file.append(SVM_forecaster5[k][1])
        for k in range(len(SVM_models)):
            SVM_Optimal_p_all_forecaster_one_file.append(SVM_forecaster6[k][4])
            SVM_Optimal_q_all_forecaster_one_file.append(SVM_forecaster6[k][5])
            SVM_Optimal_C_all_forecaster_one_file.append(SVM_forecaster6[k][3])
            SVM_MSE_collect_one_file.append(SVM_forecaster6[k][0])
            SVM_QL_collect_one_file.append(SVM_forecaster6[k][1])

        SVM_Optimal_p_q_all_forecaster_one_file =  pd.concat( [pd.Series(SVM_Optimal_p_all_forecaster_one_file),
                                                              pd.Series(SVM_Optimal_q_all_forecaster_one_file),
                                                               pd.Series(SVM_Optimal_C_all_forecaster_one_file) ], axis=1)
        One_Forecaster_titles = [time + ' SVM forecaster1',time + ' KernelSVM_rbf forecaster1',time + ' KernelSVM_sigmoid forecaster1']
        SVM_Model_names = [ One_Forecaster_titles[0],One_Forecaster_titles[1],One_Forecaster_titles[2],
                One_Forecaster_titles[0].replace("1", "2"),One_Forecaster_titles[1].replace("1", "2"),One_Forecaster_titles[2].replace("1", "2"),
                One_Forecaster_titles[0].replace("1", "3"),One_Forecaster_titles[1].replace("1", "3"),One_Forecaster_titles[2].replace("1", "3"),
                One_Forecaster_titles[0].replace("1", "4"),One_Forecaster_titles[1].replace("1", "4"),One_Forecaster_titles[2].replace("1", "4"),
                One_Forecaster_titles[0].replace("1", "5"),One_Forecaster_titles[1].replace("1", "5"),One_Forecaster_titles[2].replace("1", "5"),
                One_Forecaster_titles[0].replace("1", "6"),One_Forecaster_titles[1].replace("1", "6"),One_Forecaster_titles[2].replace("1", "6")]
        SVM_Optimal_p_q_all_forecaster_one_file.insert(0 ,'0',SVM_Model_names)

        SVM_Optimal_p_q_all_forecaster_one_file.columns = ['Model','Optimal p_'+name,'Optimal q_'+name, 'Optimal margin C_'+name]
        SVM_Optimal_p_q_all_forecaster_one_file.to_csv(name+'Optimal_C_p_q_all_forecaster_all_SVM_'+time + '.csv')


        SVM_MSE_QL_df_one_file  = pd.concat( [pd.Series(SVM_MSE_collect_one_file), pd.Series(SVM_QL_collect_one_file)], axis=1)
        SVM_MSE_QL_df_one_file.insert(0 ,'0',SVM_Model_names)
        SVM_MSE_QL_df_one_file.columns = ['Model','MSE_'+name, 'QL_'+name]
        SVM_MSE_QL_df_one_file.to_csv(name+'MSE_QL_SVM_'+time + '.csv')

        # generating ln(SE) plots for SVM and Kernel SVM Models
        SVM_ln_SE_all_forecaster_one_file = []
        for k in range(len(SVM_models)):
            SVM_ln_SE_all_forecaster_one_file.append(SVM_forecaster1[k][2])
        for k in range(len(SVM_models)):
            SVM_ln_SE_all_forecaster_one_file.append(SVM_forecaster2[k][2])
        for k in range(len(SVM_models)):
            SVM_ln_SE_all_forecaster_one_file.append(SVM_forecaster3[k][2])
        for k in range(len(SVM_models)):
            SVM_ln_SE_all_forecaster_one_file.append(SVM_forecaster4[k][2])
        for k in range(len(SVM_models)):
            SVM_ln_SE_all_forecaster_one_file.append(SVM_forecaster5[k][2])
        for k in range(len(SVM_models)):
            SVM_ln_SE_all_forecaster_one_file.append(SVM_forecaster6[k][2])

        SVM_ln_SE_collect_df_one_file = pd.concat( SVM_ln_SE_all_forecaster_one_file, axis=1)
        SVM_ln_SE_collect_df_one_file.columns = SVM_Model_names
        SVM_ln_SE_collect_df_one_file.set_index(test_sample.index)
        SVM_ln_SE_collect_df_one_file.plot(kind='line', figsize=(15, 10)).legend(loc='lower left')
        plt.xlabel("Years")
        plt.ylabel("ln(SE)")
        plt.title(name+' ln(SE)')
        # plt.show()
        plt.savefig(name + ' ' + time + ' SVM and Kernel SVM Models ln(SE).png')


        # output MSE and QL in to csv files for logistic regression
        LR_Optimal_p_forecaster_3_4_one_file = [ LogisticRegression_forecaster3[3],LogisticRegression_forecaster4[3] ]
        LR_Optimal_q_forecaster_3_4_one_file = [ LogisticRegression_forecaster3[4],LogisticRegression_forecaster4[4] ]
        LR_Optimal_p_q_forecaster_3_4_one_file =  pd.concat( [pd.Series(LR_Optimal_p_forecaster_3_4_one_file),
                                                              pd.Series(LR_Optimal_q_forecaster_3_4_one_file)], axis=1)

        LR_Optimal_p_q_forecaster_3_4_one_file.insert(0 ,'0',[time + ' Logistic Regression forecaster3',time + ' Logistic Regression forecaster4'])
        LR_Optimal_p_q_forecaster_3_4_one_file.columns = ['Model','Optimal p_'+name, 'Optimal q_'+name]
        LR_Optimal_p_q_forecaster_3_4_one_file.to_csv(name+'Optimal_p_q_forecaster_3_4__logisticRegression_'+time + '.csv')

        LR_MSE_collect_one_file = [LogisticRegression_forecaster1[0],LogisticRegression_forecaster2[0],LogisticRegression_forecaster3[0],
                                LogisticRegression_forecaster4[0],LogisticRegression_forecaster5[0],LogisticRegression_forecaster6[0]]
        LR_QL_collect_one_file = [LogisticRegression_forecaster1[1],LogisticRegression_forecaster2[1],LogisticRegression_forecaster3[1],
                                LogisticRegression_forecaster4[1],LogisticRegression_forecaster5[1],LogisticRegression_forecaster6[1]]
        LR_MSE_QL_df_one_file  = pd.concat( [pd.Series(LR_MSE_collect_one_file), pd.Series(LR_QL_collect_one_file)], axis=1)
        LR_MSE_QL_df_one_file.insert(0 ,'0',[time + ' Logistic Regression forecaster1',time + ' Logistic Regression forecaster2',
                                             time + ' Logistic Regression forecaster3',time + ' Logistic Regression forecaster4',
                                              time + ' Logistic Regression forecaster5',time + ' Logistic Regression forecaster6'])
        LR_MSE_QL_df_one_file.columns = ['Model','MSE_'+name, 'QL_'+name]
        LR_MSE_QL_df_one_file.to_csv(name+'MSE_QL_logisticRegression_'+time + '.csv')


        # generating ln(SE) plots for logistic regression
        LR_ln_SE_collect_df_one_file = pd.concat( [LogisticRegression_forecaster1[2],LogisticRegression_forecaster2[2],
                                                   LogisticRegression_forecaster3[2],LogisticRegression_forecaster4[2],
                                                   LogisticRegression_forecaster5[2],LogisticRegression_forecaster6[2]], axis=1)
        LR_ln_SE_collect_df_one_file.columns = [time + ' Logistic Regression forecaster1',time + ' Logistic Regression forecaster2',
                                                time + ' Logistic Regression forecaster3',time + ' Logistic Regression forecaster4',
                                                 time + ' Logistic Regression forecaster5',time + ' Logistic Regression forecaster6']
        LR_ln_SE_collect_df_one_file.set_index(test_sample.index)
        LR_ln_SE_collect_df_one_file.plot(kind='line', figsize=(15, 10)).legend(loc='lower left')
        plt.xlabel("Years")
        plt.ylabel("ln(SE)")
        plt.title(name+' ln(SE)')
        # plt.show()
        plt.savefig(name + ' ' + time + ' Logistic Regression ln(SE).png')




print("Complete")