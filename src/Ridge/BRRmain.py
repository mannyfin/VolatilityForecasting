import numpy as np
import BayesianRegression as brr


def BRR(train_set, test_set, warmup_period):
    brr_mse_list = []
    brr_ql_list = []
    brr_lnSE_list = []
    brr_PredVol_list = []
    brr_optimal_n_list = []
    brr_optimal_log_alpha1_list = []
    brr_optimal_log_alpha2_list = []
    brr_optimal_log_lambda1_list = []
    brr_optimal_log_lambda2_list = []

    """
           Bayesian Ridge Regression
    """
    print(str('-') * 36 + "\n\nPerforming Bayesian Ridge Regression\n\n")
    print("Training ... \n")
    # Current status: Working code for train set
    # TODO vary alphas and lamdas while holding n = 9
    # TODO vary both n and lamdas and alphas
    for n in range(1, 11):
        for alpha1 in np.exp(np.arange(-17, -3, 1)):
            for alpha2 in np.exp(np.arange(-17, -3, 1)):
                for lamda1 in np.exp(np.arange(-17, -3, 1)):
                    for lamda2 in np.exp(np.arange(-17, -3, 1)):
                        MSE, QL, ln_SE, b, c = brr.bayes_ridge_reg(train_set, n, warmup_period, alpha_1=alpha1,
                                                                   alpha_2=alpha2,
                                                                   lambda_1=lamda1, lambda_2=lamda2, test=False)
                        brr_mse_list.append(MSE)

                        print("BRR MSE for n=" + str(n) + " is: " + str(MSE))

    n = brr_mse_list.index(min(brr_mse_list)) + 1  # add one because index starts at zero
    print("The smallest n for BRR is n=" + str(n))
    print('\nTesting ...\n')