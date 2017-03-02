from SEplot import se_plot as SE
import pandas as pd
from Performance_Measure import *
from arch import arch_model
import numpy as np


class GarchModel(object):
    """
            garch(p,q) model generalized for daily, weekly and monthly data

    :return: weekly_ret
    """
    def __init__(self, df):
        self.df = df

    def garch_pq(ret, p, q,lags):

        # from arch import arch_model
        # The default set of options produces a model with a constant mean, GARCH(1,1) conditional variance and normal errors.
        garchpq = arch_model(ret, p=p, q=q, lags=lags)
        res = garchpq.fit(update_freq=0, disp='on', show_warning=True)
        forecasts = res.forecast()

        return np.sqrt(forecasts.variance['h.1'][len(ret)-1])

    def garch_pq_mse(data, Timedt, ret, p, q, lags, warmup_period, filename):

        """
        :param data: observed data
        :param ret: growing window returns
        :param Timedt:"Daily","Weekly", "Monthly"
        :param p: p=1
        :param q: q=1
        :param lags: lags=0
        :param warmup_period: 3 <= warmup_period <= len(ret)-1
        :return: MSE, QL

        """

        import numpy as np

        if Timedt == "Daily":
            TimeScaling = np.sqrt(252)
        elif Timedt == "Weekly":
            TimeScaling = np.sqrt(52)
        elif Timedt == "Monthly":
            TimeScaling = np.sqrt(12)

        dates = data['Date']
        # data['Volatility_Time'] = data['Volatility_Time'].multiply(100)
        tempdata2 = data * 100
        ret = ret * 100

        garch_pq_forecasts = []
        observed = []
        # dates=[]
        for i in range(len(ret)-warmup_period+1):
        # for i in range(len(ret)-warmup_period):
            garch_pq_forecasts.append(GarchModel.garch_pq(ret[0:warmup_period+i-1], p, q, lags))

            observed.append(tempdata2['Volatility_Time'][warmup_period + i-1])


        garch_pq_forecasts = pd.Series(garch_pq_forecasts)
        observed=pd.Series(observed)
        dates = dates[warmup_period-1:(len(ret) + 1)]

        # Instantiate the class and pass the mean_se and quasi_likelihood functions
        Performance_ = PerformanceMeasure()
        MSE = Performance_.mean_se(observed=observed/100, prediction=garch_pq_forecasts/100 * TimeScaling)
        QL = Performance_.quasi_likelihood(observed=observed/100, prediction=garch_pq_forecasts/100 * TimeScaling)

        label = str(filename)+" "+str(Timedt) + " SE: GARCH("+str(p)+","+str(q)+") "
        SE(observed/100, garch_pq_forecasts/100, dates, function_method=label)

        return MSE, QL


    def arch_q(ret, q, lags):

            archq = arch_model(ret, q=q, lags=lags, vol="Arch")
            res = archq.fit(update_freq=0, disp='off', show_warning=False)
            forecasts = res.forecast()

            return np.sqrt(forecasts.variance['h.1'][len(ret)-1])

    def arch_q_mse(data,  Timedt, ret, q, lags, warmup_period, filename):
        """
        :param filename: the file name
        :param data: observed data
        :param ret: growing window returns
        :param Timedt:"Daily","Weekly", "Monthly"
        :param q: q=1
        :param lags: lags=0
        :param warmup_period: 3 <= warmup_period <= len(ret)-1
        :return: MSE, QL
        """
        import matplotlib.pyplot as plt

        if Timedt == "Daily":
            TimeScaling = np.sqrt(252)
        elif Timedt == "Weekly":
            TimeScaling = np.sqrt(52)
        elif Timedt == "Monthly":
            TimeScaling = np.sqrt(12)
        dates = data['Date']

        tempdata = data*100
        ret = ret * 100

        arch_q_forecasts = []
        observed = []
        # dates = []
        for i in range(len(ret)-warmup_period+1):

            arch_q_forecasts.append(GarchModel.arch_q(ret[0:warmup_period+i-1], q, lags))
            observed.append(tempdata['Volatility_Time'][warmup_period + i-1])

        arch_q_forecasts = pd.Series(arch_q_forecasts)
        observed = pd.Series(observed)
        dates =dates[warmup_period-1:(len(ret) + 1)]

        # Instantiate the class and pass the mean_se and quasi_likelihood functions
        Performance_ = PerformanceMeasure()
        MSE = Performance_.mean_se(observed=observed/100, prediction=arch_q_forecasts/100 * TimeScaling)
        QL = Performance_.quasi_likelihood(observed=observed/100, prediction=arch_q_forecasts/100 * TimeScaling)

        label = str(filename)+" "+str(Timedt) + " SE: ARCH(" + str(q) + ") "

        SE(observed/100, arch_q_forecasts/100, dates, function_method=label)

        return MSE, QL