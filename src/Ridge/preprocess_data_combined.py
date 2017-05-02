import read_in_files as rd
import Volatility as vol
import pandas as pd
import split_data as sd
import os

filenames = ['AUDUSD.csv', 'CADUSD.csv',  'CHFUSD.csv', 'EURUSD.csv', 'GBPUSD.csv', 'NOKUSD.csv', 'NZDUSD.csv']

df_list = []
daily_vol_result_list=[]
name_list=[]

for name in filenames:

    print("Reading file: " + str(name))

    df, df_single_day = rd.read_in_files(name, day=1)
    name = name.split('.')[0]
    name_list.append(name)

    # daily_vol_result is the entire vol dataset
    daily_vol_result, daily_ret, daily_vol_zeroes, daily_ret_zeroes = vol.time_vol_calc(df_single_day)

    daily_vol_result_list.append(daily_vol_result.Volatility_Time)

df = pd.concat(daily_vol_result_list,axis=1).dropna()
df.insert(loc=0, column='Date', value=daily_vol_result.Date)


# make sure this saves to the dir that you want
os.chdir('Data')
df.to_csv('combined_vols.csv')


train_set, test_set = sd.split_data(dataframe=daily_vol_result, idx=910, reset_index=False)

train_set.to_csv('train_set_comb.csv')
test_set.to_csv('test_set_comb.csv')

print('Complete')