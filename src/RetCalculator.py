class RetCalculator(object):
    """
    Creates a RetCalculator class to calculate daily, weekly, or monthly returns
    """
    def __init__(self, df):
        self.df = df

    @staticmethod
    def daily_ret(df):
        """
                df here is daily 5-minute data

        :return: daily_ret
        """
        import numpy as np
        daily_ret = np.log(df.Close)[len(df.Close)-1] - np.log(df.Close[0])

        return daily_ret

    def weekly_ret(df):
        """
                df here is weekly 5-minute data

        :return: weekly_ret
        """
        import numpy as np
        weekly_ret = np.log(df.Close)[len(df.Close)-1] - np.log(df.Close[0])

        return weekly_ret

    def monthly_ret(df):
        """
                df here is monthly 5-minute data

        :return: monthly_ret
        """
        import numpy as np
        monthly_ret = np.log(df.Close)[len(df.Close)-1] - np.log(df.Close[0])

        return monthly_ret

