from Performance_Measure import *

class PastAsPresent(object):

    def tn_pred_tn_plus_1(data):

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
        prediction = data['Volatility_Time'][:-1]
        # second vec is the true values to compare
        observed = data['Volatility_Time'][1:]

        # Instantiate the class and pass the mean_se and quasi_likelihood functions
        Performance_ = PerformanceMeasure()
        MSE = Performance_.mean_se(observed=observed, prediction=prediction)
        QL = Performance_.quasi_likelihood(observed=observed, prediction=prediction)

        """ return a plot of the Squared error"""
        SE(observed, prediction)
        plt.title("Squared Error PastAsPresent (" + str(1) + ") - XYZCHANGETHIS Volatility")
        # plt.show()

        return MSE, QL

