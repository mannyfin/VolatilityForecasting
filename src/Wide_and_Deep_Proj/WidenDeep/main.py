from read_in_files import read_in_files
from Volatility import *
import numpy as np


filenames = ['AUDUSD.csv']
namelist = []

for filename in filenames:
    df, df_single_day, df_single_week, df_single_month = read_in_files(filename)
    name = filename.split('.')[0]
    namelist.append(name)
    print("Running file: " + str(name))
    time_vol_result, time_ret_result, time_vol_result_zeroes, time_ret_result_zeroes, da_rest = time_vol_calc(df_single_day)

    # squared volatility:
    da_rest['sqr_vol'] = ((da_rest.Volatility_Time)**2).astype('float64')

    # High-low spread and High-low spread ratio:
    da_rest['hl_spread'] = (da_rest.High - da_rest.Low).astype('float64')
    da_rest['hl_spread_ratio'] = ((da_rest.High - da_rest.Low)/da_rest.Low).astype('float64')

    # Open-close spread:
    da_rest['oc_spread'] = (da_rest.Close - da_rest.Open).astype('float64')

    # return categorical
    da_rest['ret_comp'] = da_rest.Return_Time.diff().astype('float64')
    da_rest.ret_comp = da_rest.ret_comp.astype('float64').apply(np.sign).fillna(value=0)

    # volatility categorical
    da_rest['vol_comp'] = da_rest.Volatility_Time.diff().astype('float64')
    da_rest.vol_comp = da_rest.vol_comp.astype('float64').apply(np.sign).fillna(value=0)

    # squared volatility categorical: (AKA variance)
    da_rest['sqr_vol_comp'] = da_rest.sqr_vol.diff().astype('float64')
    da_rest.sqr_vol_comp = da_rest.sqr_vol_comp.astype('float64').apply(np.sign).fillna(value=0)

    #  High-low spread and High-low spread ratio categorical
    da_rest['hl_spread_comp'] = da_rest.hl_spread.diff().astype('float64')
    da_rest.hl_spread_comp = da_rest.hl_spread_comp.astype('float64').apply(np.sign).fillna(value=0)

    da_rest['hl_spread_ratio_comp'] = da_rest.hl_spread_ratio.diff().astype('float64')
    da_rest.hl_spread_ratio_comp = da_rest.hl_spread_ratio_comp.astype('float64').apply(np.sign).fillna(value=0)

    # Open close spread categorical
    da_rest['oc_spread_comp'] = da_rest.oc_spread.diff().astype('float64')
    da_rest.oc_spread_comp = da_rest.oc_spread_comp.astype('float64').apply(np.sign).fillna(value=0)

    # delete the first row of the data frame as the first value of each categorical input is 0
    da_rest_new = da_rest.ix[1:]

    print("hi")

    # TODO combine da_rest with WideAndDeep.py:
    # TODO look at these functions: i.e. lines 71 to 96 in WideAndDeep.py
    # TODO remove first row with zero or nans