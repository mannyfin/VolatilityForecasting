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
    # return categorical
    da_rest['ret_comp'] = da_rest.Return_Time.diff().astype('float64')
    da_rest.ret_comp = da_rest.ret_comp.astype('float64').apply(np.sign).fillna(value=0)
    # volatility categorical
    da_rest['vol_comp'] = da_rest.Volatility_Time.diff().astype('float64')
    da_rest.vol_comp = da_rest.vol_comp.astype('float64').apply(np.sign).fillna(value=0)

    # High low spread:
    da_rest['hl_spr_rat'] = ((da_rest.High - da_rest.Low)/da_rest.Low).astype('float64')
    da_rest['oc_spread'] = (da_rest.Close - da_rest.Open).astype('float64')
    print("hi")

    # TODO combine da_rest with WideAndDeep.py:
    # TODO look at these functions: i.e. lines 71 to 96 in WideAndDeep.py