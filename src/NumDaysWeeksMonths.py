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
        month =str(pd.to_datetime(df.Date.unique(), format='%m/%d/%Y')[i].month)

        dow = pd.to_datetime(df.Date.unique(), format='%m/%d/%Y')[i].dayofweek
        # the code below finds the days in a year and the day of the week
        if year in days_weeks_months.keys():
            days_weeks_months[year]['days'] += 1
            # if days_weeks_months[]
            if dow < days_weeks_months[year]['dow']:
                days_weeks_months[year]['weeks'] += 1
            days_weeks_months[year]['dow'] = dow
            if int(month) > pd.to_datetime(df.Date.unique(), format='%m/%d/%Y')[i-1].month:
                days_weeks_months[year]['months'] += 1
        else:
            days_weeks_months[year] = {'days': 1, 'weeks': 0, 'months': 1, 'dow': dow}

    # 2008
    # startyear = df_single_month[0]['year'].unique()[0]
    #  i goes from 0 to 60 months
    # for i in range(len(df_single_month)):
    #     if

    # Find the number of days per year by cycling through the dictionary keys

    num_days_per_year = [days_weeks_months[str(year_value)]['days'] for count, year_value in enumerate(df['year'].unique())]
    num_weeks_per_year = [days_weeks_months[str(year_value)]['weeks'] for count, year_value in enumerate(df['year'].unique())]
    num_months_per_year = [days_weeks_months[str(year_value)]['months'] for count, year_value in enumerate(df['year'].unique())]


    return days_weeks_months, num_days_per_year, num_weeks_per_year, num_months_per_year
