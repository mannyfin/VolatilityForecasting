from read_in_files import read_in_files
from Volatility import *
from returnvoldf import retvoldf
import matplotlib.pyplot as plt
from PastAsPresent import *
from linear_regression import *
import numpy as np
import pandas as pd
from PPT import *


filenames = ['AUDUSD.csv']
# filenames = ['AUDUSD.csv', 'CADUSD.csv',  'CHFUSD.csv', 'EURUSD.csv', 'GBPUSD.csv', 'NOKUSD.csv', 'NZDUSD.csv']

dailyvol_zeroes= pd.DataFrame()
weeklyvol_zeroes= pd.DataFrame()
dailyret_zeroes= pd.DataFrame()
weeklyret_zeroes= pd.DataFrame()
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
    daily_vol_result, daily_ret, daily_vol_zeroes, daily_ret_zeroes = time_vol_calc(df_single_day)
    weekly_vol_result, weekly_ret, weekly_vol_zeroes, weekly_ret_zeroes = time_vol_calc(df_single_week)

    # be careful of the underscore, _
    dailyvol_zeroes = pd.concat([dailyvol_zeroes, daily_vol_zeroes], axis=1)
    dailyvol_zeroes.rename(columns={'Volatility_Time': name}, inplace=True)

    weeklyvol_zeroes = pd.concat([weeklyvol_zeroes, weekly_vol_zeroes], axis=1)
    weeklyvol_zeroes.rename(columns={'Volatility_Time': name}, inplace=True)

    # be careful of the underscore, _
    dailyret_zeroes = pd.concat([dailyret_zeroes, daily_ret_zeroes], axis=1)
    weeklyret_zeroes = pd.concat([weeklyret_zeroes, weekly_ret_zeroes], axis=1)

    # dailyret_zeroes = pd.concat([dailyret_zeroes, daily_ret_zeroes['Return_Time']], axis=1)
    dailyret_zeroes.rename(columns={'Return_Time': name}, inplace=True)
    weeklyret_zeroes.rename(columns={'Return_Time': name}, inplace=True)

    # running PastAsPresent model
    PastAsPresent = PastAsPresent.tn_pred_tn_plus_1(data=daily_vol_result)
    # PastAsPresent = PastAsPresent.tn_pred_tn_plus_1(data=daily_vol_result[1:])

    # running linear regression models with number of regressors as 1,3,5 and 10
    warmup_daily = 400 # for linear regression
    LinearReg_1 = lin_reg(data=daily_vol_result, n=1, warmup_period=warmup_daily)
    LinearReg_3 = lin_reg(data=daily_vol_result, n=3, warmup_period=warmup_daily)
    LinearReg_5 = lin_reg(data=daily_vol_result, n=5, warmup_period=warmup_daily)
    LinearReg_10 = lin_reg(data=daily_vol_result, n=10, warmup_period=warmup_daily)

    # running linear regression models with the optimal number of regressors
    # TODO: get the training and test sample and use the optimal n obtained in linear_regression.py on the test smaple
    train_sample_daily_FirstThreeHalfYrs =
    test_sample_daily_LastOneHalfYrs =
    LinearReg_opt = lin_reg(data=daily_vol_result, n=OptNumReg, warmup_period=warmup_daily)



    "create returnvoldf for Kernel Ridge Regression"
    "We want to test daily and weekly data"
    preprocess_daily_from_v, test_sample_daily_from_v, train_sample_daily_from_v = retvoldf(daily_ret, daily_vol_result, v)
    preprocess_weekly_from_v,test_sample_weekly_from_v, train_sample_weekly_from_v = retvoldf(weekly_ret, weekly_vol_result, v)

    for time in ["Daily","Weekly"]:
        if time == "Daily":
            train_sample = train_sample_daily_from_v
            test_sample = test_sample_daily_from_v
        elif time =="Weekly":
            train_sample = train_sample_weekly_from_v
            test_sample = test_sample_weekly_from_v



print("hi")

""" For LASSO with ridge regression"""
# create daily_vol_combined combining daily volatilities of all 7 currency pairs, with date as the index
# drop zero volatilities
daily_vol_combine = dailyvol_zeroes[(dailyvol_zeroes != 0).all(1)]
dailydates = daily_vol_combine.Date.loc[:, ~daily_vol_combine.Date.columns.duplicated()]
daily_vol_combined = daily_vol_combine.set_index(dailydates.Date,drop=True)
# drop duplicate columns of date
daily_vol_combined.drop('Date', axis=1, inplace=True)
daily_vol_combined=daily_vol_combined.apply(pd.to_numeric)

# create weekly_vol_combined combining weekly volatilities of all 7 currency pairs, with date as the index
# drop zero volatilities
weekly_vol_combine = weeklyvol_zeroes[(weeklyvol_zeroes != 0).all(1)]
weeklydates = weekly_vol_combine.Date.loc[:, ~weekly_vol_combine.Date.columns.duplicated()]
weekly_vol_combined = weekly_vol_combine.set_index(weeklydates.Date,drop=True)
# drop duplicate columns of date
weekly_vol_combined.drop('Date', axis=1, inplace=True)
weekly_vol_combined=weekly_vol_combined.apply(pd.to_numeric)


