class RetCalculator(object):
    """
    Creates a RetCalculator class to calculate daily, weekly, or monthly returns
    """
    def __init__(self, df):
        self.df = df

    @staticmethod

    def ret(df):
        """
                df here contains daily 5-minute data

        :return: daily_ret
        """
        import numpy as np
        ret = np.mean(np.log(df.Close) - np.log(df.Close.shift(len(df.Close)-1)))
        return ret


    def daily_ret_df(df, df_single_day, num_days_per_year):
        # import pandas as pd
        # num_days_per_year = [NumDays2008,NumDays2009,NumDays2010,NumDays2011,NumDays2012,NumDays2013]

        daily_rets = []
        for i in range(num_days_per_year[0]):
            daily_rets.append(RetCalculator.ret(df_single_day[i]))
        for i in range(num_days_per_year[1]):
            daily_rets.append(RetCalculator.ret(df_single_day[i + num_days_per_year[0]]))
        for i in range(num_days_per_year[2]):
            daily_rets.append(RetCalculator.ret(df_single_day[i + num_days_per_year[0] + num_days_per_year[1]]))
        for i in range(num_days_per_year[3]):
            daily_rets.append(RetCalculator._ret(
                df_single_day[i + num_days_per_year[0] + num_days_per_year[1] + num_days_per_year[2]]))
        for i in range(num_days_per_year[4]):
            daily_rets.append(RetCalculator.ret(df_single_day[i + num_days_per_year[0] + num_days_per_year[1] +
                                                                 num_days_per_year[2] + num_days_per_year[3]]))
        for i in range(num_days_per_year[5]):
            daily_rets.append(RetCalculator.ret(df_single_day[i + num_days_per_year[0] + num_days_per_year[1] +
                                                                 num_days_per_year[2] + num_days_per_year[3] +
                                                                 num_days_per_year[4]]))

        d = {'Date': df.Date.unique(), 'Return_Daily': daily_rets}
        daily_ret_result = pd.DataFrame(d)

        return daily_ret_result

    def monthly_ret_df(df, df_single_month):
        # import pandas as pd
        # num_days_per_year = [NumDays2008,NumDays2009,NumDays2010,NumDays2011,NumDays2012,NumDays2013]

        monthly_rets = []
        for i in range(len(df_single_month)):
            monthly_rets.append(RetCalculator.ret(df_single_month[i]))

        d = {'Date': df.Date.unique(), 'Return_Monthly': monthly_rets}
        monthly_ret_result = pd.DataFrame(d)

        return monthly_ret_result
