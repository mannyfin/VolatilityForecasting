
""" 
1. Read in files, 
2. Calculate Vol 
3. split data into train and test set
 return train and test set
 """

import read_in_files as rd
import Volatility as vol
import split_data as sd


def preprocess(name):

    # initialize some lists

    print("Running file: " + str(name))
    #  reads in the files and puts them into dataframes, returns a dataframe called df
    df, df_single_day = rd.read_in_files(name, day=1)
    name = name.split('.')[0]
    # name.append(name)


    # daily_vol_result is the entire vol dataset
    daily_vol_result, daily_ret, daily_vol_zeroes, daily_ret_zeroes = vol.time_vol_calc(df_single_day)

    #  Split the dataset into train and test set
    #  909 is break point for train/test
    train_set, test_set = sd.split_data(dataframe=daily_vol_result, idx=910, reset_index=False)

    return train_set, test_set