
import pandas as pd
import numpy as np
import KernelRidgeRegression as krr
from makedirs import makedirs


def KRR(train_set, test_set, warmup_period, name,n_seq, dictlist, param_range=np.exp(np.arange(-17, -3, 1))):

    makedirs('Ridge//Results', 'BayesianRidgeRegression', name=name)

    if KRR.__name__ not in dictlist:

        dictlist[KRR.__name__] = dict()

        dictlist[KRR.__name__]['krr1_mse_list'] = []
        dictlist[KRR.__name__]['krr1_ql_list'] = []
        dictlist[KRR.__name__]['krr1_lnSE_list'] = []
        dictlist[KRR.__name__]['krr1_PredVol_list'] = []
        dictlist[KRR.__name__]['krr_optimal_log_alpha1_list'] = []
        dictlist[KRR.__name__]['krr_optimal_log_alpha2_list'] = []
        dictlist[KRR.__name__]['krr_optimal_log_lambda1_list'] = []
        dictlist[KRR.__name__]['krr_optimal_log_lambda2_list'] = []
        #
    """
           Kernel Ridge Regression
    """

    print(str('-') * 34 + "\n\nPerforming Kernel Ridge Regression\n\n")
    print("Training ... \n")
    # Current status: Working code for train set
    # TODO vary alphas and lamdas while holding n = 9
    # TODO vary a whole bunch of stuff
    kernels=['linear','gaussian', 'polynomial', 'sigmoid', 'rbf', 'laplacian' ]  #  chi2

    for kernel in kernels:
        for n in range(1,11):
            MSE, QL, ln_SE = krr.kernel_ridge_reg(train_set, n, warmup_period, alpha=1,coef0=1, degree=3, kernel=kernel, test=False)
            krr_mse_list.append(MSE)
            print("KRR MSE for n=" + str(n) + " and kernel=" + kernel + " is: " + str(MSE))

        n = krr_mse_list.index(min(krr_mse_list)) + 1  # add one because index starts at zero
        print("The smallest n for KRR is n=" + str(n) + " for kernel = " + kernel)
        print("\nTraining ... \n")

    print('\nTesting ...\n')

    # feel free to put a breakpoint in the line below...
    print('hi')