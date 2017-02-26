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

        ret = np.mean(np.log(df.Close) - np.log(df.Close.shift(len(df.Close)-1)))
        return ret

    # @staticmethod
    def daily_vol(df, num_days_per_year):
        """
        :pnum_days_per__yearam df: df in the input is a data frame containing data of a pnum_days_per__yearticulnum_days_per__year day
        :pnum_days_per__yearam n: n is the number of trading days in a pnum_days_per__yearticulnum_days_per__year yenum_days_per__year
        :return: annualized_daily_vol
        """
        # TODO  use n as sqrt(288 * numdays of year)

        vol = np.std(np.log(df.Close) - np.log(df.Close.shift(1)))
        
        annualized_daily_vol = vol * np.sqrt(num_days_per_year)
        return annualized_daily_vol
    
    
def time_vol_calc(df, df_single_time, num_days_per_year):

    # num_days_per_year = [NumDays2008,NumDays2009,NumDays2010,NumDays2011,NumDays2012,NumDays2013]
    # TODO LOOP THIS AND ALSO MAKE MORE USE OF THE VOLATILITY CLASS

    time_vols = []
    time_rets = []

    for i in range(num_days_per_year[0]):
        time_vols.append(Volatility.daily_vol(df_single_time[i], num_days_per_year[0]))
        time_rets.append(Volatility.ret(df_single_time[i]))

    for i in range(num_days_per_year[1]):
        time_vols.append(Volatility.daily_vol(df_single_time[i + num_days_per_year[0]], num_days_per_year[1]))
        time_rets.append(Volatility.ret(df_single_time[i + num_days_per_year[0]]))

    for i in range(num_days_per_year[2]):
        time_vols.append(Volatility.daily_vol(df_single_time[i + num_days_per_year[0] + num_days_per_year[1]], num_days_per_year[2]))
        time_rets.append(Volatility.ret(df_single_time[i + num_days_per_year[0] + num_days_per_year[1]]))

    for i in range(num_days_per_year[3]):
        time_vols.append(Volatility.daily_vol(df_single_time[i + num_days_per_year[0] + num_days_per_year[1] + num_days_per_year[2]], num_days_per_year[3]))
        time_rets.append(Volatility.ret(
            df_single_time[i + num_days_per_year[0] + num_days_per_year[1] + num_days_per_year[2]]))

    for i in range(num_days_per_year[4]):
        time_vols.append(Volatility.daily_vol(df_single_time[i + num_days_per_year[0] + num_days_per_year[1] + num_days_per_year[2] + num_days_per_year[3]], num_days_per_year[4]))
        time_rets.append(Volatility.ret(df_single_time[i + num_days_per_year[0] + num_days_per_year[1] +
                                                          num_days_per_year[2] + num_days_per_year[3]]))

    for i in range(num_days_per_year[5]):
        time_vols.append(Volatility.daily_vol(df_single_time[i + num_days_per_year[0] + num_days_per_year[1] + num_days_per_year[2] + num_days_per_year[3] + num_days_per_year[4]], num_days_per_year[5]))
        time_rets.append(Volatility.ret(df_single_time[i + num_days_per_year[0] + num_days_per_year[1] +
                                                          num_days_per_year[2] + num_days_per_year[3] +
                                                          num_days_per_year[4]]))

    # TODO reformat code to include dvol and dret as a single DataFrame
    # gives an error for df.Date.unique() when using week and month because the num vals are not the same
    dvol = {'Date': [df_single_time[i]['Date'][df_single_time[i]['Date'].first_valid_index()] for i in range(0, len(df_single_time))], 'Volatility_Daily': time_vols}
    dret = {'Date': [df_single_time[i]['Date'][df_single_time[i]['Date'].first_valid_index()] for i in range(0, len(df_single_time))], 'Return_Daily': time_rets}
    len(time_vols)  # outputs 260
    len(time_rets)   # outputs 260
    len([df_single_time[i]['Date'][df_single_time[i]['Date'].first_valid_index()] for i in range(0, len(df_single_time))])  # outputs 263

    time_vol_result = pd.DataFrame(dvol)
    time_ret_result = pd.DataFrame(dret)

    inters_ret = time_ret_result.query('Return_Daily == 0').index.values
    inters_vol = time_vol_result.query(('Volatility_Daily ==0')).index.values
    """
    # this line below removes days where the vols were zero
    # time_vol_result = time_vol_result.query('Volatility_Daily != 0')
    """
    comparison_array = list(set(inters_ret).intersection(inters_vol))

    time_vol_result = time_vol_result.drop(time_vol_result.index[comparison_array])
    time_ret_result = time_ret_result.drop(time_ret_result.index[comparison_array])

    return time_vol_result, time_ret_result



    # def monthly_df(df, df_single_month):
    #     # num_days_per_year = [NumDays2008,NumDays2009,NumDays2010,NumDays2011,NumDays2012,NumDays2013]
    #
    #     monthly_rets = []
    #     for i in range(len(df_single_month)):
    #         monthly_rets.append(RetCalculator.ret(df_single_month[i]))
    #
    #     d = {'Index': range(1,len(df_single_month)+1), 'Return_Monthly': monthly_rets}
    #     monthly_ret_result = pd.DataFrame(d)
    #
    #     return monthly_ret_result