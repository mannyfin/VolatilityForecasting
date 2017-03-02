import matplotlib.pyplot as plt


def count(func):
    def wrapper(*args, **kwargs):

        if wrapper.temp%8 == 0:    # executed every time the wrapped function is called
            wrapper.counter += 1
        wrapper.temp+=1

        return func(*args, **kwargs)
    # wrapper.counter = 0         # executed only once in decorator definition time
    wrapper.counter = 1
    wrapper.temp = wrapper.counter     # executed only once in decorator definition time
    return wrapper


@count
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
    SE = (y_fit.ravel() - y.ravel()) ** 2


    # reshape will be deprecated. the line below is not necessarily the correct one.
    # SE = (y_fit.values.reshape(len(y), 1) - y.values.reshape(len(y), 1)) ** 2
    # plt.figure(n)
    plt.figure(se_plot.counter, figsize=(12,7))


    ts2 = pd.DataFrame({'SE': np.ravel(np.log(SE))})
    # may need to reset index..
    dates = dates.reset_index()
    dates = dates.Date

    ts2['Date'] = pd.DataFrame(dates)


    plt.gcf()
    plt.plot(ts2['Date'], ts2['SE'], label=function_method)



    plt.xlabel("Years")
    plt.ylabel("ln(SE)")


