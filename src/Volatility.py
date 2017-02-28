import numpy as np
import pandas as pd

class Volatility(object):
    """
    Creates a Volatility class, which can be used to calculate daily, weekly, or monthly volatility
    """
    # TODO FIX WEEKLY, AND MONTHLY VOLATILITY N'S
    def __init__(self, df):
        self.df = df

    def ret(df):
        """
                df here contains daily 5-minute data

        :return: daily_ret
        """

        ret = np.log(df['Close'][df.index[-1]]/df['Close'][df.index[0]])
        # ret = np.sum(np.log(df.Close) - np.log(df.Close.shift(1)))
        return ret


    # @staticmethod
    def vol_scaling(df):
        num_days_per_year = 313*288
        # this was previously daily_vol

        """
        :pnum_days_per__yearam df: df in the input is a data frame containing data of a pnum_days_per__yearticulnum_days_per__year day
        :pnum_days_per__yearam n: n is the number of trading days in a pnum_days_per__yearticulnum_days_per__year yenum_days_per__year
        :return: annualized_vol_scaling
        """
        # TODO  use n as sqrt(288 * numdays of year)

        vol = np.std(np.log(df.Close) - np.log(df.Close.shift(1)))
        
        annualized_vol_scaling = vol * np.sqrt(num_days_per_year)
        return annualized_vol_scaling
    
    
def time_vol_calc(df_single_time):

    # num_days_per_year = 313
    # TODO LOOP THIS AND ALSO MAKE MORE USE OF THE VOLATILITY CLASS

    time_vols = []
    time_rets = []

    for i in range(len(df_single_time)):
        time_vols.append(Volatility.vol_scaling(df_single_time[i]))
        time_rets.append(Volatility.ret(df_single_time[i]))

    # TODO reformat code to include dvol and dret as a single DataFrame
    # gives an error for df.Date.unique() when using week and month because the num vals are not the same
    dvol = {'Date': [df_single_time[i]['Date'][df_single_time[i]['Date'].first_valid_index()] for i in range(0, len(df_single_time))], 'Volatility_Daily': time_vols}
    dret = {'Date': [df_single_time[i]['Date'][df_single_time[i]['Date'].first_valid_index()] for i in range(0, len(df_single_time))], 'Return_Daily': time_rets}

    time_vol_result = pd.DataFrame(dvol)
    time_ret_result = pd.DataFrame(dret)

    inters_ret = time_ret_result.query('Return_Daily == 0').index.values
    # essentially you will only ever pass through this if statement if there are zero return values
    if inters_ret.size > 0:
        inters_vol = time_vol_result.query(('Volatility_Daily == 0')).index.values
        """
        # this line below removes days where the vols were zero
        # time_vol_result = time_vol_result.query('Volatility_Daily != 0')
        """
        # bug point, maybe get an error if c
        comparison_array = list(set(inters_ret).intersection(inters_vol))

        time_vol_result = time_vol_result.drop(time_vol_result.index[comparison_array])
        time_ret_result = time_ret_result.drop(time_ret_result.index[comparison_array])

    return time_vol_result, time_ret_result
