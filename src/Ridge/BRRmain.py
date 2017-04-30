"""
only do BRR1
"""
import pandas as pd
import numpy as np
import BayesianRegression as brr
from makedirs import makedirs


def BRR(train_set, test_set, warmup_period, name, count, n_seq, dictlist, param_range):
    """
    growing window bayesian ridge regression
    :param train_set: 
    :param test_set: 
    :param warmup_period: warmup period
    :param name: name of file w/o .csv
    :param n_seq: for BRR1, this is a constant. for BRR2 it is a list 
    :param dictlist: dictionary of outputs from test set
    :param param_range: 4-tuple of parameter ranges to explore
    :return: dictionary list
    """

<<<<<<< HEAD
    makedirs('Ridge//Results', 'BayesianRidgeRegression', name=name)
=======
    makedirs('Ridge//Results', '(d) BayesianRidgeRegression2', name=name)
>>>>>>> 6afa2dc12bcf67508c9b7a99b8ab3e06f5b9fb45

    if BRR.__name__ not in dictlist:

        dictlist[BRR.__name__] = dict()

        dictlist[BRR.__name__]['brr1_mse_list'] = []
        dictlist[BRR.__name__]['brr1_ql_list'] = []
        dictlist[BRR.__name__]['brr1_lnSE_list'] = []
        dictlist[BRR.__name__]['brr1_PredVol_list'] = []
        dictlist[BRR.__name__]['brr_optimal_log_alpha1_list'] = []
        dictlist[BRR.__name__]['brr_optimal_log_alpha2_list'] = []
        dictlist[BRR.__name__]['brr_optimal_log_lambda1_list'] = []
        dictlist[BRR.__name__]['brr_optimal_log_lambda2_list'] = []
    #
    # brr_mse_list = []
    # brr_ql_list = []
    # brr_lnSE_list = []
    # brr_PredVol_list = []
    # brr_optimal_n_list = []
    # brr_optimal_log_alpha1_list = []
    # brr_optimal_log_alpha2_list = []
    # brr_optimal_log_lambda1_list = []
    # brr_optimal_log_lambda2_list = []

    """
           Bayesian Ridge Regression
    """
    print(str('-') * 36 + "\n\nPerforming Bayesian Ridge Regression\n\n")
    print("Training ... \n")
    # Current status: Working code for train set
    # TODO vary alphas and lamdas while holding n = 9
    # TODO vary both n and lamdas and alphas

    # for n in range(1, 11):   #this is brr2
    # for n in range(n_seq):    #this is brr1
    #     for alpha1 in np.exp(np.arange(-17, -3, 1)):
    #         for alpha2 in np.exp(np.arange(-17, -3, 1)):
    #             for lamda1 in np.exp(np.arange(-17, -3, 1)):
    #                 for lamda2 in np.exp(np.arange(-17, -3, 1)):
    # for n in range(1,n_seq):    #this is brr2
    n = n_seq  #just including n =const
    mselists, alpha1list, alpha2list, lamda1list, lamda2list = [], [], [], [], []

    for alpha1 in param_range[0]:
        for alpha2 in param_range[1]:
            for lamda1 in param_range[2]:
                for lamda2 in param_range[3]:
                    MSE, QL, ln_SE, b, c = brr.bayes_ridge_reg(train_set, n, warmup_period, alpha_1=alpha1,
                                                               alpha_2=alpha2,
                                                               lambda_1=lamda1,
                                                               lambda_2=lamda2)
                    # brr_mse_list.append(MSE)
                    mselists.append(MSE)

                    # i just use these to quickly find the optimal params later
                    alpha1list.append(alpha1)
                    alpha2list.append(alpha2)
                    lamda1list.append(lamda1)
                    lamda2list.append(lamda2)

                    print("BRR_n=" + str(n) + " MSE:  " + str(MSE)+"   ; QL: "+str(QL)+"  ; alpha1: "+str(alpha1) +
                          "  ; alpha2:" + str(alpha2) + "  ; lamda1:  " + str(lamda1) + "  ; lamda2:" + str(lamda2))

    # find the best combo of alpha1, alpha2, lamda1, lamda2
    # n = brr_mse_list.index(min(brr_mse_list)) + 1  # add one because index starts at zero
    # Only need the line below for BRR2
    val_idx = mselists.index(min(mselists))  # add one because index starts at zero

    opt_alpha1 = alpha1list[val_idx]
    opt_alpha2 = alpha2list[val_idx]
    opt_lamda1 = lamda1list[val_idx]
    opt_lamda2 = lamda2list[val_idx]

    dictlist[BRR.__name__]['brr_optimal_log_alpha1_list'].append(alpha1list[val_idx])
    dictlist[BRR.__name__]['brr_optimal_log_alpha2_list'].append(alpha2list[val_idx])
    dictlist[BRR.__name__]['brr_optimal_log_lambda1_list'].append(lamda1list[val_idx])
    dictlist[BRR.__name__]['brr_optimal_log_lambda2_list'].append(lamda2list[val_idx])
    # print("The smallest n for BRR is n=" + str(n))

    #
    print('\nTesting ...\n')
    # BRR test set. Use the entire training set as the fit for the test set. See code in BRR.
    MSE_BRR1_test, QL_BRR1_test, ln_SE_BRR1_test, PredVol_BRR1_test, b_BRR1_test, c_BRR1_test = brr.bayes_ridge_reg(
        train_set, n, warmup_period ,alpha_1=opt_alpha1, alpha_2=opt_alpha2, lambda_1=opt_lamda1, lambda_2=opt_lamda2,
        test=(True, test_set))

    dictlist[BRR.__name__]['brr1_mse_list'].append(MSE_BRR1_test)
    dictlist[BRR.__name__]['brr1_ql_list'].append(QL_BRR1_test)
    dictlist[BRR.__name__]['brr1_lnSE_list'].append(ln_SE_BRR1_test)
    dictlist[BRR.__name__]['brr1_PredVol_list'].append(PredVol_BRR1_test)

    # print(str(name) + " BRR1(" + str(lr_optimal_n_list_benchmark[count]) + ")" + "log_lamdba_" + str(
    #     rr1_optimal_log_lambda) + " test MSE: " + str(MSE_RR1_test) + "; test QL: " + str(QL_RR1_test))

    print(str(name) + " BRR1(" + str(n) + ")" + " test MSE: " + str(MSE_BRR1_test) + "; test QL: " + str(QL_BRR1_test))

    brr1_lnse = dictlist[BRR.__name__]['brr1_lnSE_list'][count]
    brr1_predvol = dictlist[BRR.__name__]['brr1_PredVol_list'][count]

    brr1_lnSE_list_df = pd.DataFrame(np.array([brr1_lnse]), index=["brr1_lnSE"]).transpose()
    brr1_PredVol_list_df = pd.DataFrame(np.array([brr1_predvol]), index=["brr1_PredVol"]).transpose()
    brr1_lnSE_list_df.to_csv(str(name)+" brr1_lnSE.csv")
    brr1_PredVol_list_df.to_csv(str(name)+" brr1_PredVol.csv")

    return dictlist
