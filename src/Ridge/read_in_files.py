
def read_in_files(file_names):
    """
    1. Read the csv files to memory into a pandas dataframe with pd.read_csv
    2. separate the df into year, month, and date objects
    3. It also chunks the data by single day
    """
    import os
    import pandas as pd
    # change working directory to where the files are located
    # startdir = os.chdir(os.path.join(os.getenv('userprofile'), 'Desktop\\FIN580\\Homework1'))
    try:
        os.chdir(os.path.join(os.getenv('userprofile'), 'Desktop/FIN580/Homework1/VolatilityForecasting/src/Data'))
    except ValueError:
            raise ValueError("No possible directory found")

    file1 = pd.read_csv(file_names)
    # file1 = pd.read_csv(file_names, parse_dates=[['Date', 'Time']])
    os.chdir('..')

    
    df = pd.DataFrame(file1)
    # comment out line below
    df.Date = pd.to_datetime(df.Date, format='%m/%d/%Y')

    df['temp'] = df['Date'].astype(str) + ' ' + df['Time']
    df.temp = pd.to_datetime(df.temp, infer_datetime_format=True)
    df.temp = df.temp + pd.offsets.Hour(8)


    # sunday 4pm to Friday 4pm
    # # this is a timestamp obj
    df['year'], df['month'] = df['Date'].dt.year, df['Date'].dt.month
    df['date'] = df['Date'].dt.day
    df['week'] = df['Date'].dt.week

    """
    these three lines below chunk the data by dates
    """

    # df_single_day = []
    # for group in df.groupby(df.Date, sort=False):
    #     df_single_day.append(group[1])


    g = df.groupby(df['temp'].dt.normalize())
    df_single_day = []
    df_single_week = []
    w = 0
    for group in g:
        if len(group[1]) > 1:
            df_single_day.append(group[1])
            if (len(df_single_day)-3)%5 is 0 and (len(df_single_day)-3) != 0:
                # 4th day to
                df_single_week.append(pd.concat(df_single_day[3+w*5:]))

                w+=1

    # df_single_week = []
    # for group in df.groupby(['week', 'year'], sort=False):
    #     df_single_week.append(group[1])

    df_single_month = []
    for group in df.groupby(['month', 'year'], sort=False):
        df_single_month.append(group[1])

    # these are the chunks of the daily, weekly, and monthly data
    return df, df_single_day, df_single_week, df_single_month

"""
this code may be useful
kw = lambda x: x.isocalendar()[1];
kw_year = lambda x: str(x.year) + ' - ' + str(x.isocalendar()[1])
grouped = df.groupby([df['Date'].map(kw_year)], sort=False, as_index=False)
"""