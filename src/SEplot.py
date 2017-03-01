import matplotlib.pyplot as plt


# def count(func):
#     def wrapper(*args, **kwargs):
#         wrapper.counter += 1    # executed every time the wrapped function is called
#         return func(*args, **kwargs)
#     wrapper.counter = 0         # executed only once in decorator definition time
#     return wrapper
#
#
# @count
def se_plot(y, y_fit, dates, function_method):
    """
    :param y: source data
    :param y_fit: fit from LR
    :param n: the num of trailing days. This is an [int]
    :return:
    """

    # def se_plot(x, y, y_fit1):
    import numpy as np
    import pandas as pd

    # Squared error
    # SE = (y_fit.reshape(len(y), 1) - y.reshape(len(y), 1)) ** 2
    SE = (y_fit.ravel() - y.ravel()) ** 2


    # reshape will be deprecated. the line below is not necessarily the correct one.
    # SE = (y_fit.values.reshape(len(y), 1) - y.values.reshape(len(y), 1)) ** 2
    # plt.figure(n)
    # TODO add DPI
    # plt.figure(se_plot.counter, figsize=(12,5))
    plt.figure(1, figsize=(12,5))
    # nplogse = pd.DataFrame._from_arrays(np.log(SE), index=dates1, columns=['SE'])
    # dates = dates.dt.to_period(freq='m')

    # TODO maybe make this line below a dataframe
    # ts = pd.Series(np.ravel(np.log(SE)), index=pd.date_range(dates[dates.first_valid_index()], periods=(dates.last_valid_index()- dates.first_valid_index()+1)))
    ts2 = pd.DataFrame({'SE': np.ravel(np.log(SE))})
    # ts2['Date'] = pd.DataFrame(dates[dates.first_valid_index(): (dates.last_valid_index() - dates.first_valid_index() + 1)])
    # may need to reset index..
    dates = dates.reset_index()
    dates = dates.Date

    ts2['Date'] = pd.DataFrame(dates)

    # ts.plot()
    ax = plt.plot(ts2['Date'], ts2['SE'], label=function_method)
    plt.hold(True)
    # plt.plot(dates, np.log(SE))


    # TODO make this plot vs months/years etc.
    plt.xlabel("Years")
    plt.ylabel("ln(SE)")


