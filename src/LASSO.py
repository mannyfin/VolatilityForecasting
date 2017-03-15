from VAR import x_mat_t_n_qp as xw
import numpy as np

def lassofn(y, xw, alpha = 0.1):
    q = 9
    p = 3
    n = p * 100
    LogRV_df = np.log(daily_vol_combined)
    xw(LogRV_df,q, p, t, n)
