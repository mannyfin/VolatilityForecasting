"""



                    import pandas as pd
                    import numpy as np

                    file1 = pd.read_csv('AUDUSD.csv')

                    file1 = pd.DataFrame(file1)
                    file1.Date = pd.to_datetime(file1.Date, format='%m/%d/%Y')

                    df = file1

                    # this is a timestamp obj
                    df.Date = pd.to_datetime(df.Date, format='%m/%d/%Y')
                    df['year'], df['month'] = df['Date'].dt.year, df['Date'].dt.month
                    df['date'] = df['Date'].dt.day


                    # these three lines below chunk the data by dates

                    DFList = []
                    for group in file1.groupby(file1.Date, sort=False):
                        DFList.append(group[1])

                    Complete
"""
# this gives the date to compare
DFList[1].Date[DFList[1].index[0]]
# gives the year
DFList[2].Date[DFList[2].index[0]].year
# gives total number of unique days
len(df.Date.unique())

"""
count the number of days in each year
"""

# NumDays2008 = 0
# NumDays2009 = 0
# NumDays2010 = 0
# NumDays2011 = 0
# NumDays2012 = 0
# NumDays2013 = 0
#
# for i in range(len(df.Date.unique())):
#    if pd.to_datetime(df.Date.unique(), format='%m/%d/%Y')[i].year==2008:
#        NumDays2008 += 1
#    elif pd.to_datetime(df.Date.unique(), format='%m/%d/%Y')[i].year==2009:
#        NumDays2009 += 1
#    elif pd.to_datetime(df.Date.unique(), format='%m/%d/%Y')[i].year==2010:
#        NumDays2010 += 1
#    elif pd.to_datetime(df.Date.unique(), format='%m/%d/%Y')[i].year==2011:
#        NumDays2011 += 1
#    elif pd.to_datetime(df.Date.unique(), format='%m/%d/%Y')[i].year==2012:
#        NumDays2012 += 1
#    elif pd.to_datetime(df.Date.unique(), format='%m/%d/%Y')[i].year==2013:
#        NumDays2013 += 1
#
# NumDays =[NumDays2008,NumDays2009,NumDays2010,NumDays2011,NumDays2012,NumDays2013]


"""
create a function for calculating annualized daily volatility
"""


def DailyVol(df, n):
    # df in the input is a data frame containing data of a particular day
    # n is the number of trading days in a particular year
    Vol = np.std(np.log(df.Close) - np.log(df.Close.shift(1)))
    Annualized_DailyVol = Vol * np.sqrt(n)
    return Annualized_DailyVol


"""
create a function for calculating annualized weekly volatility
"""


def WeeklyVol(df):
    # df in the input is a data frame containing data of a particular week
    # we approximate the number of weeks in a year as 52
    Vol = np.std(np.log(df.Close) - np.log(df.Close.shift(1)))
    Annualized_WeeklyVol = Vol * np.sqrt(52)
    return Annualized_WeeklyVol


"""
create a function for calculating annualized weekly volatility
"""


def MonthlyVol(df):
    # df in the input is a data frame containing data of a particular month
    # the number of weeks in a year is 12
    Vol = np.std(np.log(df.Close) - np.log(df.Close.shift(1)))
    Annualized_MonthlyVol = Vol * np.sqrt(12)
    return Annualized_MonthlyVol


"""
get the number of days in each year and make it into an array
"""

"""

                                    def NumDaysWeeksMonths():
                                        asdf = {}
                                        for i in range(len(df.Date.unique())):
                                            year = str(pd.to_datetime(df.Date.unique(), format='%m/%d/%Y')[i].year)
                                            dow = pd.to_datetime(df.Date.unique(), format='%m/%d/%Y')[i].dayofweek
                                            if year in asdf.keys():
                                                asdf[year]['days'] += 1
                                                if dow < asdf[year]['dow']:
                                                    asdf[year]['weeks'] += 1
                                                asdf[year]['dow'] = dow
                                            else:
                                                asdf[year] = {'days': 1, 'weeks': 0, 'dow': dow}
                                        return asdf

                                        COMPLETED
"""

'''NumDays2008 = 0
NumDays2009 = 0
NumDays2010 = 0
NumDays2011 = 0
NumDays2012 = 0
NumDays2013 = 0

for i in range(len(df.Date.unique())):
    if pd.to_datetime(df.Date.unique(), format='%m/%d/%Y')[i].year==2008:
        NumDays2008 += 1
    elif pd.to_datetime(df.Date.unique(), format='%m/%d/%Y')[i].year==2009:
        NumDays2009 += 1
    elif pd.to_datetime(df.Date.unique(), format='%m/%d/%Y')[i].year==2010:
        NumDays2010 += 1
    elif pd.to_datetime(df.Date.unique(), format='%m/%d/%Y')[i].year==2011:
        NumDays2011 += 1
    elif pd.to_datetime(df.Date.unique(), format='%m/%d/%Y')[i].year==2012:
        NumDays2012 += 1
    elif pd.to_datetime(df.Date.unique(), format='%m/%d/%Y')[i].year==2013:
        NumDays2013 += 1

NumDays =[NumDays2008,NumDays2009,NumDays2010,NumDays2011,NumDays2012,NumDays2013]
'''

"""
create a data frame to contain annualized daily volatility for each day
"""

"""
                                                asdf = NumDaysWeeksMonths()
                                                NumDays = [asdf['2008']['days'], asdf['2009']['days'], asdf['2010']['days'],
                                                           asdf['2011']['days'], asdf['2012']['days'], asdf['2013']['days']]
                                                COMPLETED
"""
"""
                                                def DailyVolDF(df, ar):
                                                    # ar = [NumDays2008,NumDays2009,NumDays2010,NumDays2011,NumDays2012,NumDays2013]
                                                    # df is the input, which can be file1...file9
                                                    DailyVols = []
                                                    for i in range(ar[0]):
                                                        DailyVols.append(DailyVol(DFList[i], ar[0]))
                                                    for i in range(ar[1]):
                                                        DailyVols.append(DailyVol(DFList[i + ar[0]], ar[1]))
                                                    for i in range(ar[2]):
                                                        DailyVols.append(DailyVol(DFList[i + ar[0] + ar[1]], ar[2]))
                                                    for i in range(ar[3]):
                                                        DailyVols.append(DailyVol(DFList[i + ar[0] + ar[1] + ar[2]], ar[3]))
                                                    for i in range(ar[4]):
                                                        DailyVols.append(DailyVol(DFList[i + ar[0] + ar[1] + ar[2] + ar[3]], ar[4]))
                                                    for i in range(ar[5]):
                                                        DailyVols.append(DailyVol(DFList[i + ar[0] + ar[1] + ar[2] + ar[3] + ar[4]], ar[5]))

                                                    d = {'Date': df.Date.unique(), 'Volatility_Daily': DailyVols}
                                                    NewDailyVolDF = pd.DataFrame(d)

                                                    return NewDailyVolDF

                                                    COMPLETED

"""
"""
Today’s vol as a forecast of tomorrow’s volatiltiy - using daily volatility
"""


def linear_regression_daily_vol_1():
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    lr = LinearRegression()
    data = np.asarray(DailyVolDF(file1, NumDays)['Volatility_Daily'])
    # use today’s vol as a forecast of tomorrow’s volatiltiy
    # y represents tomorrow's volatility and x represents today's volatility
    x, y = data[:-1], data[1:]
    x = x.reshape(len(x), 1)
    lr.fit(x, y)
    b = lr.coef_[0]
    c = lr.intercept_
    y_fit1 = b * x + c
    MSE1 = mean_squared_error(y, y_fit1)
    print("MSE1 is " + str(MSE1))
    print("intercept is " + str(c))
    print("slope is " + str(b))


    '''
    using the formula MSE Correct!
    '''

    #    (1/len(y))*np.sum((y_fit1.reshape(len(y),1)-y.reshape(len(y),1))**2)
    SE = (y_fit1.reshape(len(y), 1) - y.reshape(len(y), 1)) ** 2
    plt.plot(SE)
    plt.title("Squared Error LR(1) - Daily Volatility")
    plt.xlabel("t")
    plt.ylabel("SE")
    ### TODO change x-axis to time series

    '''
    using the formula QL
    '''
    value = y_fit1.reshape(len(y), 1) / y.reshape(len(y), 1)
    Ones = np.ones(len(y))

    (1 / len(y)) * (np.sum(value - np.log(value) - Ones.reshape(len(y), 1)))

    plt.scatter(x, y, color='black')
    plt.plot(x, y_fit1, color='blue', linewidth=3)
    plt.xticks(())
    plt.yticks(())
    return (c, b, MSE1)

"""
Today’s vol as a forecast of tomorrow’s volatiltiy - using daily volatility
"""


def linear_regression_daily_vol_3():
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    data = np.asarray(DailyVolDF(file1, NumDays)['Volatility_Daily'])
    # use past 3 volatilties to predict tomorrow’s volatiltiy
    # y represents tomorrow's volatility;
    # x1, x2 and x3 represent past 3 volatilties

    x1, x2, x3, y = data[2:-1], data[1:-2], data[:-3], data[3:]
    x1 = x1.reshape(len(x1), 1)
    x2 = x2.reshape(len(x2), 1)
    x3 = x3.reshape(len(x3), 1)
    x = np.c_[x1, x2, x3]
    lr.fit(x, y)
    b1 = lr.coef_[0]
    b2 = lr.coef_[1]
    b3 = lr.coef_[2]

    c3 = lr.intercept_
    # IMPORT ERROR CALC FUNCTION
    #    plot it as in the example at http://scikit-learn.org/
    plt.scatter(y, y, color='black')
    #    plt.plot(x, regr.predict(x), color='blue', linewidth=3)
    plt.plot(x, (b1 * x1 + b2 * x2 + b3 * x3 + c3), color='blue', linewidth=3)
    plt.xticks(())
    plt.yticks(())

    return (c3, b1, b2, b3)


#
# def linear_regression_daily_vol_1():
##    from sklearn import datasets, linear_model
##    import matplotlib.pyplot as plt
#    import pandas as pd
#    import statsmodels.formula.api as sm
#
#    # IMPORT ERROR CALC FUNCTION
#    # use today’s vol as a forecast of tomorrow’s volatiltiy
#
#    # y represents tomorrow's volatility and x represents today's volatility
#
#    x = DailyVolDF(file1,NumDays)[:-1]
#    x = x['Volatility_Daily']
#    y = DailyVolDF(file1,NumDays)[1:]
#    y = y['Volatility_Daily']
#
#
#    df = pd.DataFrame({"A": y, "B": x})
#    result = sm.ols(formula="A ~ B", data=df).fit()
#    print(result.params)
#    print(result.summary())
#
##    regr = linear_model.LinearRegression()
##    regr.fit(x.reshape(len(x), 1), y.reshape(len(y), 1))
#
#    # plot it as in the example at http://scikit-learn.org/
#    fit1=plt.figure(1)
#    plt.scatter(x, y,  color='black')
##    plt.plot(x, regr.predict(x), color='blue', linewidth=3)
#    plt.plot(x, (result.params.B*x+result.params.Intercept), color='blue', linewidth=3)
#    plt.xticks(())
#    plt.yticks(())
#    return fit1



"""
test the DailyVolDF(df,ar) function
"""
# df = file1
# ar = NumDays
# DailyVolDF(file1,NumDays)


"""
create a data frame to contain annualized monthly volatility for each month
"""


# def MonthlyVolDF(df):
#     # df is the input, which can be file1...file9
#
#     # Chunk the data by months to get DFList_Mon using the function you are working on
#
#
#     # format the datetime into "year-month" format
#     MonthYear = pd.to_datetime(df.Date.unique()).map(lambda x: x.strftime('%Y-%m'))
#     UniqueMonthYear = np.unique(MonthYear)
#
#     MonthlyVols = []
#     for i in range(len(UniqueMonthYear)):
#         MonthlyVols.append(MonthlyVol(DFList_Mon[i]))
#
#     d = {'Date': UniqueMonthYear, 'Volatility_Monthly': MonthlyVols}
#     NewMonthlyVolDF = pd.DataFrame(d)
#
#     return NewMonthlyVolDF


'''
Export dataframe to Excel
'''
exportFileName = 'linear_regression_daily_vol_1'
# sample_name = 'linear_regression_daily_vol_1'
df_date = DailyVolDF(file1, NumDays)

import os

os.chdir('C:\\Users\\Manolis1\\Desktop\\FIN580\\Homework1')
engine = 'xlsxwriter'
filename = "linear_regression_daily_vol_1_%s.xlsx" % engine
writer = pd.ExcelWriter(filename, engine=engine)
# df.to_excel("out.xls")
df.to_excel(writer)

# writer = pd.ExcelWriter(exportFileName)
##df.to_excel(writer, sheet_name=sample_name, startrow=0, startcol=1)
# df_date.to_excel(writer, sheet_name=sample_name, startrow=0, startcol=0, index=False, engine=io.excel.xlsm.writer)
# writer.save()




"""
Today’s vol as a forecast of tomorrow’s volatiltiy - using daily volatility
"""

print("hi")
# def linear_regression_daily_vol_1():
##    from sklearn import datasets, linear_model
#    import matplotlib.pyplot as plt
#    import pandas as pd
##    import statsmodels.formula.api as sm
##    from pandas.stats.api import ols
#    from sklearn.linear_model import LinearRegression
#    from sklearn.metrics import mean_squared_error
#    lr = LinearRegression()
#    data = np.asarray(DailyVolDF(file1,NumDays)['Volatility_Daily'])
#    # use today’s vol as a forecast of tomorrow’s volatiltiy
#    # y represents tomorrow's volatility and x represents today's volatility
#    x, y = data[:-1], data[1:]
#    x =x.reshape(len(x), 1)
#    lr.fit(x, y)
#    b=lr.coef_[0]
#    c=lr.intercept_
#    y_fit1 = b*x+c
#    MSE1 = mean_squared_error(y, y_fit1)
#    print("MSE1 is "+ str(MSE1))
#    print("intercept is "+str(c))
#    print("slope is "+str(b))
#

