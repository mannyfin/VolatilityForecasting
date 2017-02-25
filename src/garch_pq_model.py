from SEplot import se_plot as SE
import pandas as pd

class garch_model(object):
    """
            garch

    :return: weekly_ret
    """
    def __init__(self, df):
        self.df = df

    def garch_pq(ret, p, q,lags):

        from arch import arch_model
        # The default set of options produces a model with a constant mean, GARCH(1,1) conditional variance and normal errors.
        garchpq = arch_model(ret,p=p, q=q,lags=lags)
        res = garchpq.fit(update_freq=1)
        forecasts = res.forecast()

        return forecasts.variance['h.1'][1]

    def garch_pq_mse(data,ret,p,q,lags):
        from sklearn.metrics import mean_squared_error as mse
        import matplotlib.pyplot as plt


        garch_pq_forecasts = []
        for i in range(len(ret)-2):
            garch_pq_forecasts.append(garch_model.garch_pq(ret[i:(i+lags+1)], p, q,lags))
        # observed daily vol
        observed = data['Volatility_Daily'][2:]
        garch_pq_forecasts = pd.Series(garch_pq_forecasts)
        output = mse(observed, garch_pq_forecasts)
        SE(observed,garch_pq_forecasts)
        print("PastAsPresent MSE is :" + str(MSE_oneday))

        plt.show()
        return output




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

