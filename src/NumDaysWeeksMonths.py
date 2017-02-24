def NumDaysWeeksMonths(df):
    """
    Takes the input df and finds the days, weeks, and months in the pd.dataframe
    :param df:
    :return days_weeks_months, num__days_per__year:
    """
    import pandas as pd
    days_weeks_months = {}
    for i in range(len(df.Date.unique())):
        year = str(pd.to_datetime(df.Date.unique(), format='%m/%d/%Y')[i].year)
        #  TODO FIX DAY OF THE WEEK, IT APPEARS TO BE INCORRECT
        dow = pd.to_datetime(df.Date.unique(), format='%m/%d/%Y')[i].dayofweek
        if year in days_weeks_months.keys():
            days_weeks_months[year]['days'] += 1
            if dow < days_weeks_months[year]['dow']:
                days_weeks_months[year]['weeks'] += 1
            days_weeks_months[year]['dow'] = dow
        else:
            days_weeks_months[year] = {'days': 1, 'weeks': 0, 'dow': dow}

    # Find the number of days per year by cycling through the dictionary keys
    num_days_per__year = [days_weeks_months[year_value]['days'] for year_value in days_weeks_months.keys()]

    return days_weeks_months, num_days_per__year
