import pandas as pd
import numpy as np

class RetCalculator(object):
    """
    Creates a RetCalculator class to calculate daily, weekly, or monthly returns
    """
    def __init__(self, df):
        self.df = df

    def ret(df):
        """
                df here contains daily 5-minute data

        :return: daily_ret
        """

        ret = np.mean(np.log(df.Close) - np.log(df.Close.shift(len(df.Close)-1)))
        return ret


    def daily_ret_df(df, df_single_day, num_days_per_year):
        # num_days_per_year = [NumDays2008,NumDays2009,NumDays2010,NumDays2011,NumDays2012,NumDays2013]

        daily_rets = []
        for i in range(num_days_per_year[0]):
            daily_rets.append(RetCalculator.ret(df_single_day[i]))
        for i in range(num_days_per_year[1]):
            daily_rets.append(RetCalculator.ret(df_single_day[i + num_days_per_year[0]]))
        for i in range(num_days_per_year[2]):
            daily_rets.append(RetCalculator.ret(df_single_day[i + num_days_per_year[0] + num_days_per_year[1]]))
        for i in range(num_days_per_year[3]):
            daily_rets.append(RetCalculator.ret(
                df_single_day[i + num_days_per_year[0] + num_days_per_year[1] + num_days_per_year[2]]))
        for i in range(num_days_per_year[4]):
            daily_rets.append(RetCalculator.ret(df_single_day[i + num_days_per_year[0] + num_days_per_year[1] +
                                                                 num_days_per_year[2] + num_days_per_year[3]]))
        for i in range(num_days_per_year[5]):
            daily_rets.append(RetCalculator.ret(df_single_day[i + num_days_per_year[0] + num_days_per_year[1] +
                                                                 num_days_per_year[2] + num_days_per_year[3] +
                                                                 num_days_per_year[4]]))

        d = {'Date': df.Date.unique(), 'Return_Daily': daily_rets}

        # need to make a comparison between Daily Return and Daily Volatility. For our test sample, only one data point/
        # in daily volatility is zero, but multiple data points in daily return are zero. We need to compare these and
        # remove the common elements


        daily_ret_result = pd.DataFrame(d)

        return daily_ret_result

    def monthly_ret_df(df, df_single_month):
        # num_days_per_year = [NumDays2008,NumDays2009,NumDays2010,NumDays2011,NumDays2012,NumDays2013]

        monthly_rets = []
        for i in range(len(df_single_month)):
            monthly_rets.append(RetCalculator.ret(df_single_month[i]))

        d = {'Index': range(1,len(df_single_month)+1), 'Return_Monthly': monthly_rets}
        monthly_ret_result = pd.DataFrame(d)

        return monthly_ret_result


