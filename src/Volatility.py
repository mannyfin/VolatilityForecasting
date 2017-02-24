class Volatility(object):
    """
    Creates a Volatility class, which can be used to calculate daily, weekly, or monthly volatility
    """
    # TODO FIX WEEKLY, AND MONTHLY VOLATILITY N'S
    def __init__(self, df):
        self.df = df
        
    # @staticmethod
    def daily_vol(df, num_days_per__yenum_days_per__year):
        """
        :pnum_days_per__yearam df: df in the input is a data frame containing data of a pnum_days_per__yearticulnum_days_per__year day
        :pnum_days_per__yearam n: n is the number of trading days in a pnum_days_per__yearticulnum_days_per__year yenum_days_per__year
        :return: annualized_daily_vol
        """
        # TODO  use n as sqrt(288 * numdays of year)
        import numpy as np
        vol = np.std(np.log(df.Close) - np.log(df.Close.shift(1)))
        
        annualized_daily_vol = vol * np.sqrt(num_days_per__yenum_days_per__year)
        return annualized_daily_vol
    
    
def daily_vol_calc(df, df_single_day, num_days_per_year):
    import pandas as pd
    # num_days_per_year = [NumDays2008,NumDays2009,NumDays2010,NumDays2011,NumDays2012,NumDays2013]
    # df is the input, which can be file1...file9
    # TODO LOOP THIS AND ALSO MAKE MORE USE OF THE VOLATILITY CLASS
    daily_vols = []
    for i in range(num_days_per_year[0]):
        daily_vols.append(Volatility.daily_vol(df_single_day[i], num_days_per_year[0]))
    for i in range(num_days_per_year[1]):
        daily_vols.append(Volatility.daily_vol(df_single_day[i + num_days_per_year[0]], num_days_per_year[1]))
    for i in range(num_days_per_year[2]):
        daily_vols.append(Volatility.daily_vol(df_single_day[i + num_days_per_year[0] + num_days_per_year[1]], num_days_per_year[2]))
    for i in range(num_days_per_year[3]):
        daily_vols.append(Volatility.daily_vol(df_single_day[i + num_days_per_year[0] + num_days_per_year[1] + num_days_per_year[2]], num_days_per_year[3]))
    for i in range(num_days_per_year[4]):
        daily_vols.append(Volatility.daily_vol(df_single_day[i + num_days_per_year[0] + num_days_per_year[1] + num_days_per_year[2] + num_days_per_year[3]], num_days_per_year[4]))
    for i in range(num_days_per_year[5]):
        daily_vols.append(Volatility.daily_vol(df_single_day[i + num_days_per_year[0] + num_days_per_year[1] + num_days_per_year[2] + num_days_per_year[3] + num_days_per_year[4]], num_days_per_year[5]))

    d = {'Date': df.Date.unique(), 'Volatility_Daily': daily_vols}
    daily_vol_result = pd.DataFrame(d)

    return daily_vol_result
