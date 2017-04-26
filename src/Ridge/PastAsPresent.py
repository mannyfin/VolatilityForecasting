from Performance_Measure import *

class PastAsPresent(object):

    def tn_pred_tn_plus_1(data):

        """
        data could be daily_vol_result obtained in main.py
        
        starting prediction point is t1
        end point is tn
        So to predict tomorrow's volatility using today's volatility compare:
        1. t0 with t1
        2. t1 with t2
        ...
        n. tn-1 with tn
    
        So you have two vectors. 1. from t0 to tn-1, and 2. from t1 to tn
    
        then do the calcs
    
        """

        # first vec is the prediction
        prediction = data['Volatility_Time'][:-1]
        # second vec is the true values to compare
        observed = data['Volatility_Time'][1:]

        # Instantiate the class and pass the mean_se and quasi_likelihood functions
        Performance_ = PerformanceMeasure()
        MSE = Performance_.mean_se(observed=observed, prediction=prediction)
        QL = Performance_.quasi_likelihood(observed=observed.astype('float64'),
                                           prediction=prediction.astype('float64'))

        # label = str(filename) + " " + str(stringinput) + " Squared Error PastAsPresent (" + str(1) + ") Volatility"
        # """ return a plot of the Squared error"""
        # SE(observed, prediction, dates,function_method=label)
        SE = [(observed.values[i] - prediction.values[i]) ** 2 for i in range(len(observed))]
        ln_SE = pd.Series(np.log(SE)) # the type of ln_SE is pandas.core.series.Series
        PredVol = prediction
        return MSE, QL,ln_SE,PredVol

