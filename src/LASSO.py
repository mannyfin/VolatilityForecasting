from VAR import *
import numpy as np

def lassofn(y, xw, alpha = 0.1, stringin):
    q = 9
    p = 3
    n = p * 100
    LogRV_df = np.log(daily_vol_combined)
    # xw(LogRV_df,q, p, t, n)
    # training
    if stringin is 'Train':
        for t in range(p,int(2/3*len(LogRV_df))-n):
            x = x_mat_t_n_qp(LogRV_df,q, p, t,n)
            y = get_y(LogRV_df,q, t,n)


    if stringin is 'Test':
        for t in range(int(2/3*len(LogRV_df))-n,len(LogRV_df)-n):
            x = x_mat_t_n_qp(LogRV_df, q, p, t, n)
            y = get_y(LogRV_df, q, t, n)