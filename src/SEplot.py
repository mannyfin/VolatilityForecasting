import matplotlib.pyplot as plt


def count(func):
    def wrapper(*args, **kwargs):
        wrapper.counter += 1    # executed every time the wrapped function is called
        return func(*args, **kwargs)
    wrapper.counter = 0         # executed only once in decorator definition time
    return wrapper


@count
def se_plot(y, y_fit):
    """
    :param y: source data
    :param y_fit: fit from LR
    :param n: the num of trailing days. This is an [int]
    :return:
    """

    # def se_plot(x, y, y_fit1):
    import numpy as np

    # Squared error
    SE = (y_fit.reshape(len(y), 1) - y.reshape(len(y), 1)) ** 2

    # reshape will be deprecated. the line below is not necessarily the correct one.
    # SE = (y_fit.values.reshape(len(y), 1) - y.values.reshape(len(y), 1)) ** 2
    # plt.figure(n)
    plt.figure(se_plot.counter)
    plt.plot(np.log(SE))
    # TODO make LR(n)

    # TODO make this plot vs months/years etc.
    plt.xlabel("t")
    plt.ylabel("ln(SE)")


