import matplotlib.pyplot as plt

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
    # plt.figure(n)
    plt.plot(SE)
    # TODO make LR(n)
    # plt.title("Squared Error LR(" + str(n) + ") - Daily Volatility" )
    plt.xlabel("t")
    plt.ylabel("SE")

    # TODO change x-axis to time series

    '''
    using the formula QL
    '''
    # TODO QL DOES NOT WORK DUE TO ZEROES IN DATA SERIES
    # value = y_fit1.reshape(len(y), 1) / y.reshape(len(y), 1)
    # Ones = np.ones(len(y))
    #
    # (1 / len(y)) * (np.sum(value - np.log(value) - Ones.reshape(len(y), 1)))

    # # this only works with single parameter LR
    # plt.scatter(x, y, color='black')
    # plt.plot(x, y_fit1, color='blue', linewidth=3)
    # plt.xticks(())
    # plt.yticks(())
