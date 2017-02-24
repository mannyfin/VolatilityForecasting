from main import daily_vol_result
from sklearn.metrics import mean_squared_error as mse
from SEplot import se_plot as SE


def part1_today_tomorrow(data):
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
    first_vector = data['Volatility_Daily'][:-1]
    # second vec is the true values to compare
    second_vector = data['Volatility_Daily'][1:]
    # mse(y_true, y pred)
    MSE_oneday = mse(second_vector, first_vector)
    print("MSE one day is :" + str(MSE_oneday))

    return MSE_oneday
