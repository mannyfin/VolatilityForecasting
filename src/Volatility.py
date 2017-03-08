import numpy as np
import pandas as pd

class Volatility(object):
    """
    Creates a Volatility class, which can be used to calculate daily, weekly, or monthly volatility
    """
    def __init__(self, df):
        self.df = df

    def ret(df):
        """
                df here contains daily/weekly/monthly 5-minute data

        :return: ret
        """

        rets = np.log(df['Close'][df.index[-1]]/df['Close'][df.index[0]])
        # rets = np.sum(np.log(df.Close) - np.log(df.Close.shift(1)))
        return rets

    def GrowWindowRet(df, initial):
        """
                df here contains daily/weekly/monthly 5-minute data
                initial determines the size of the initial growing window

        :return: GrowWindowRets
        """
        GrowWindowRets=[]
        ret = np.log(df['Close'][df.index[-1]]/df['Close'][df.index[0]])

        return GrowWindowRets

    # @staticmethod
    def vol_scaling(df):
        num_days_per_year = 252*288
        # this was previously daily_vol

        """
        :pnum_days_per__yearam df: df in the input is a data frame containing data of a pnum_days_per__yearticulnum_days_per__year day
        :pnum_days_per__yearam n: n is the number of trading days in a pnum_days_per__yearticulnum_days_per__year yenum_days_per__year
        :return: annualized_vol_scaling
        """

        vol = np.std(np.log(df.Close) - np.log(df.Close.shift(1)))
        
        annualized_vol_scaling = vol * np.sqrt(num_days_per_year)
        return annualized_vol_scaling
    
    
def time_vol_calc(df_single_time):
    import pandas as pd

    time_vols = []
    time_rets = []

    for i in range(len(df_single_time)):
        time_vols.append(Volatility.vol_scaling(df_single_time[i]))
        time_rets.append(Volatility.ret(df_single_time[i]))

    # TODO reformat code to include dvol and dret as a single DataFrame
    # gives an error for df.Date.unique() when using week and month because the num vals are not the same
    dvol = {'Date': [df_single_time[i]['Date'][df_single_time[i]['Date'].first_valid_index()] for i in range(0, len(df_single_time))], 'Volatility_Time': time_vols}
    dret = {'Date': [df_single_time[i]['Date'][df_single_time[i]['Date'].first_valid_index()] for i in range(0, len(df_single_time))], 'Return_Time': time_rets}

    time_vol_result = pd.DataFrame(dvol)
    time_ret_result = pd.DataFrame(dret)

    time_vol_result_zeroes = time_vol_result
    time_ret_result_zeroes = time_ret_result

    inters_ret = time_ret_result.query('Return_Time == 0').index.values
    # essentially you will only ever pass through this if statement if there are zero return values
    if inters_ret.size > 0:
        inters_vol = time_vol_result.query(('Volatility_Time == 0')).index.values
        """
        # this line below removes days where the vols were zero
        # time_vol_result = time_vol_result.query('Volatility_Time != 0')
        """
        # bug point, maybe get an error if c
        comparison_array = list(set(inters_ret).intersection(inters_vol))

        time_vol_result = time_vol_result.drop(time_vol_result.index[comparison_array])
        time_ret_result = time_ret_result.drop(time_ret_result.index[comparison_array])
        time_vol_result.reset_index(drop=True, inplace=True)
        time_ret_result.reset_index(drop=True, inplace=True)

    return time_vol_result, time_ret_result, time_vol_result_zeroes, time_ret_result_zeroes
