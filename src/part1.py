
def today_tomorrow(data):

    from sklearn.metrics import mean_squared_error as mse
    import matplotlib.pyplot as plt
    from SEplot import se_plot as SE
    """
    starting prediction point is t1
    end point is tn
    So to predict tomorrow's volatility using today's volatility compare:
    1. t0 with t1
    2. t1 with t2
    ...
    n. tn-1 with tn

    So you have two vectors. 1. from t0 to tn-1, and 2. from t1 to tn

    then do the calcs

    :return:
    """

    # first vec is the prediction
    prediction = data['Volatility_Daily'][:-1]
    # second vec is the true values to compare
    observed = data['Volatility_Daily'][1:]
    # mse(y_true, y pred)
    MSE_oneday = mse(observed, prediction)
    print("MSE one day is :" + str(MSE_oneday))

    """ return a plot of the Squared error"""
    SE(observed, prediction, 1)
    plt.title("Squared Error part1 (" + str(1) + ") - Daily Volatility")

    return MSE_oneday
