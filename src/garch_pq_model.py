from SEplot import se_plot as SE
import pandas as pd
from Performance_Measure import *
from arch import arch_model


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
        res = garchpq.fit(update_freq=0, disp='off', show_warning=False)
        forecasts = res.forecast()

        return forecasts.variance['h.1'][len(ret)-1]

    def garch_pq_mse(data, Timedt, ret, p, q, lags, initial):

        """
        :param ret: growing window returns
        :param Timedt:"Daily","Weekly", "Monthly"
        :param p: p=1
        :param q: q=1
        :param lags: lags=0
        :param initial: 3 <= initial <= len(ret)-1
        :return: MSE, QL

        """

        from sklearn.metrics import mean_squared_error as mse
        import matplotlib.pyplot as plt
        import numpy as np
        import numpy as np

        if Timedt == "Daily":
            TimeScaling = np.sqrt(313)
        elif Timedt == "Weekly":
            TimeScaling = np.sqrt(52)
        elif Timedt == "Monthly":
            TimeScaling = np.sqrt(12)


        garch_pq_forecasts = []
        observed =[]
        for i in range(len(ret)-initial):
            garch_pq_forecasts.append(GarchModel.garch_pq(ret[0:initial+i-1], p, q,lags))
        # observed daily vol
            observed.append(data['Volatility_Daily'][initial + i])

        #     observed.append(data['Volatility_Daily'][initial:])

        # observed = data['Volatility_Daily'][2:]
        # garch_pq_forecasts = pd.Series(garch_pq_forecasts)

        garch_pq_forecasts = pd.Series(garch_pq_forecasts)
        observed=pd.Series(observed)
        # Instantiate the class and pass the mean_se and quasi_likelihood functions
        Performance_ = PerformanceMeasure()
        MSE = Performance_.mean_se(observed=observed, prediction=garch_pq_forecasts * TimeScaling)
        QL = Performance_.quasi_likelihood(observed=observed, prediction=garch_pq_forecasts * TimeScaling)

        # output = mse(observed, garch_pq_forecasts)

        SE(observed, garch_pq_forecasts)

        plt.title(str("Daily/weekly/monthly") + "SE: GARCH("+str(p)+","+str(q)+") ")
        #TODO: change the name of plots
        # plt.show()
        return MSE, QL


    def arch_q(ret, q, lags):

            # from arch import arch_model
            # The default set of options produces a model with a constant mean, GARCH(1,1) conditional variance\
            #  and normal errors.
            archq = arch_model(ret, q=q, lags=lags, vol="Arch")
            res = archq.fit(update_freq=0, disp='off', show_warning=False)
            forecasts = res.forecast()

            return forecasts.variance['h.1'][len(ret)-1]

    def arch_q_mse(data,  Timedt, ret, q, lags,initial):
        """

        :param ret: growing window returns
        :param Timedt:"Daily","Weekly", "Monthly"
        :param q: q=1
        :param lags: lags=0
        :param initial: 3 <= initial <= len(ret)-1
        :return: MSE, QL
        """

        # from sklearn.metrics import mean_squared_error as mse
        import matplotlib.pyplot as plt

        if Timedt == "Daily":
            TimeScaling = np.sqrt(313)
        elif Timedt == "Weekly":
            TimeScaling = np.sqrt(52)
        elif Timedt == "Monthly":
            TimeScaling = np.sqrt(12)


        arch_q_forecasts = []
        observed=[]
        for i in range(len(ret)-initial):
            arch_q_forecasts.append(GarchModel.arch_q(ret[0:initial+i-1], q, lags))
            observed.append(data['Volatility_Daily'][initial + i])
        # print("hi")
        # observed daily vol
        # observed = data['Volatility_Daily'][2:]
        arch_q_forecasts = pd.Series(arch_q_forecasts)
        observed = pd.Series(observed)


        # Instantiate the class and pass the mean_se and quasi_likelihood functions
        Performance_ = PerformanceMeasure()
        MSE = Performance_.mean_se(observed=observed, prediction=arch_q_forecasts * TimeScaling)
        QL = Performance_.quasi_likelihood(observed=observed, prediction=arch_q_forecasts * TimeScaling)

        # output = mse(observed, arch_q_forecasts)

        SE(observed, arch_q_forecasts)
        plt.title(str("Daily/weekly/monthly") + "SE: ARCH(" + str(q) + ") ")

        # plt.show()
        return MSE, QL

    # def arch_q_mse(data, ret, q, lags):
    #
    #     # from sklearn.metrics import mean_squared_error as mse
    #     import matplotlib.pyplot as plt
    #
    #     arch_q_forecasts = []
    #     for i in range(len(ret) - 2):
    #         arch_q_forecasts.append(GarchModel.arch_q(ret[i:(i + lags + 1)], q, lags))
    #     # observed daily vol
    #     observed = data['Volatility_Daily'][2:]
    #     arch_q_forecasts = pd.Series(arch_q_forecasts)
    #
    #     # Instantiate the class and pass the mean_se and quasi_likelihood functions
    #     Performance_ = PerformanceMeasure()
    #     MSE = Performance_.mean_se(observed=observed, prediction=arch_q_forecasts)
    #     QL = Performance_.quasi_likelihood(observed=observed, prediction=arch_q_forecasts)
    #
    #     # output = mse(observed, arch_q_forecasts)
    #
    #     SE(observed, arch_q_forecasts)
    #     plt.title(str(lags) + " Day Lag's SE: ARCH(" + str(q) + ") ")
    #
    #     # plt.show()
    #     return MSE, QL

    # def garch_pq_mse(data, ret, p, q, lags):
    # from sklearn.metrics import mean_squared_error as mse
    # import matplotlib.pyplot as plt
    #
    # garch_pq_forecasts = []
    # for i in range(len(ret)-2):
    #     garch_pq_forecasts.append(GarchModel.garch_pq(ret[i:(i+lags+1)], p, q,lags))
    # # observed daily vol
    # observed = data['Volatility_Daily'][2:]
    # garch_pq_forecasts = pd.Series(garch_pq_forecasts)
    #
    # # Instantiate the class and pass the mean_se and quasi_likelihood functions
    # Performance_ = PerformanceMeasure()
    # MSE = Performance_.mean_se(observed=observed, prediction=garch_pq_forecasts)
    # QL = Performance_.quasi_likelihood(observed=observed, prediction=garch_pq_forecasts)
    #
    # # output = mse(observed, garch_pq_forecasts)
    #
    # SE(observed, garch_pq_forecasts)
    #
    # plt.title(str(lags) + " Day Lag's SE: GARCH("+str(p)+","+str(q)+") ")
    # # plt.show()






    # def garch_pq(ret, p, q,lags):
    #
    #     from arch import arch_model
    #     # The default set of options produces a model with a constant mean, GARCH(1,1) conditional variance and normal errors.
    #     garchpq = arch_model(ret[1:3],p=p, q=q,lags=lags)
    #     res = garchpq.fit(update_freq=1)
    #     forecasts = res.forecast()
    #     print(forecasts.variance)
    #     # print(res.summary())


    # from arch import arch_model
    # garch11 = arch_model(r, p=1, q=1)
    # res = garch11.fit(update_freq=10)
    # print(res.summary())

