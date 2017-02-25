
def read_in_files(filenames):
    """
    1. Read the csv files to memory into a pandas dataframe with pd.read_csv
    2. separate the df into year, month, and date objects
    3. It also chunks the data by single day
    """
    import os
    import pandas as pd
    # change working directory to where the files are located
    # startdir = os.chdir(os.path.join(os.getenv('userprofile'), 'Desktop\\FIN580\\Homework1'))
    os.chdir(os.path.join(os.getenv('userprofile'), 'Desktop\\FIN580\\Homework1\\VolatilityForecasting\\src'))

    file1 = pd.read_csv(filenames)

    df = pd.DataFrame(file1)
    df.Date = pd.to_datetime(df.Date, format='%m/%d/%Y')

    # this is a timestamp obj
    df['year'], df['month'] = df['Date'].dt.year, df['Date'].dt.month
    df['date'] = df['Date'].dt.day

    """
    these three lines below chunk the data by dates
    """
    df_single_day = []
    for group in df.groupby(df.Date, sort=False):
        df_single_day.append(group[1])

    return df, df_single_day
