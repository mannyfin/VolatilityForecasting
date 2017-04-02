import matplotlib.pyplot as plt


def count(func):
    def wrapper(*args, **kwargs):
        # Change the number here based on the number of files
        # may also need to change for function specific plots
        if wrapper.temp%10 == 0:    # executed every time the wrapped function is called
            wrapper.counter += 1
        wrapper.temp+=1

        return func(*args, **kwargs)
    # wrapper.counter = 0         # executed only once in decorator definition time
    wrapper.counter = 1
    wrapper.temp = wrapper.counter     # executed only once in decorator definition time
    return wrapper


@count
def se_plot(y, y_fit, dates=None, function_method=None, mode=None):
    """
    
    :param y: source data
    :param y_fit:  fit from model
    :param dates: dates for x axis plotting
    :param function_method: MSE or QL
    :param mode: choose between MSE or ln(MSE), if mode = None, then ylabel-> ln(MSE), else ylabel->MSE
    :return: 
    """

    import numpy as np
    import pandas as pd

    # Squared error

    if isinstance(y, pd.core.frame.DataFrame) & isinstance(y_fit, pd.core.frame.DataFrame):
        # this is really logSE
        SE = np.log(np.square(np.subtract(y, y_fit)))

        # this line converts to df and transposes from cols to rows
        SE = pd.DataFrame(SE)
        # adding dates
        SE=SE.join(dates.Date)
        SE=SE.set_index('Date')

        SE.plot(kind='line', figsize=(12, 7)).legend(loc='center left', bbox_to_anchor=(1, 0.5))

    else:
        SE = (y_fit.ravel() - y.ravel()) ** 2
        if mode is None: ts2 = pd.DataFrame({'SE': np.ravel(np.log(SE.astype('float64')))})
        else: ts2 = pd.DataFrame({'SE':np.ravel(SE)})
        date_c = dates.copy()
        date_c = date_c.reset_index()
        ts2['Date'] = pd.DataFrame(date_c.Date)
        # TODO FIX THIS COUNTER ISSUE
        se_plot.counter = 1
        plt.figure(se_plot.counter, figsize=(12,7))

        dates = dates.reset_index()
        dates = dates.Date

        plt.gcf()
        plt.plot(ts2['Date'].dropna(), ts2['SE'], label=function_method)

    plt.xlabel("Years")
    if mode is None: plt.ylabel("ln(SE)")
    else: plt.ylabel("MSE") 


