
"""
Linear regression. calls linear_regression.py and produces plots and outputs.
"""

import linear_regression_combined as lr
from PastAsPresent import PastAsPresent as pap
import RidgeRegression_combined as rr
import matplotlib.pyplot as plt
from makedirs import makedirs
import KernelRidgeRegression_combined as krr
from KRR_combined_append_and_csv import KRR_append_and_csv
import os
import pandas as pd
import numpy as np
import BayesianRegression_combined as brr
from makedirs import makedirs
import os


def PaP(test_set, train_set, name, dictlist, filenames=None):

    makedirs('Ridge//Results', 'PastAsPresent', name=name)

    if PaP.__name__ not in dictlist:
        dictlist[PaP.__name__] = dict()

        dictlist[PaP.__name__]['pap_mse_list'] = []
        dictlist[PaP.__name__]['pap_ql_list'] = []
        dictlist[PaP.__name__]['pap_lnSE_list'] = []
        dictlist[PaP.__name__]['pap_PredVol_list'] = []

    """
            PastAsPresent
    """
    print(str('-') * 24 + "\n\nPerforming PastAsPresent\n\n")

    # PastAsPresent -- Test sample only


    # last element of train set added...................................
    updated_test_set = pd.concat([train_set.tail(1), test_set])

    papMSE_test, papQL_test, pap_ln_SE_test ,pap_PredVol_test = pap.tn_pred_tn_plus_1(updated_test_set)
    print("Past as Present MSE: " + str(papMSE_test) + "; QL: " + str(papQL_test))

    # pap_mse_list.append(papMSE_test)
    # pap_ql_list.append(papQL_test)
    # pap_lnSE_list.append(pap_ln_SE_test)
    # pap_PredVol_list.append(pap_PredVol_test)
    dictlist[PaP.__name__]['pap_mse_list'].append(papMSE_test)
    dictlist[PaP.__name__]['pap_ql_list'].append(papQL_test)
    dictlist[PaP.__name__]['pap_lnSE_list'].append(pap_ln_SE_test)
    dictlist[PaP.__name__]['pap_PredVol_list'].append(pap_PredVol_test)

    paplnse = dictlist[PaP.__name__]['pap_lnSE_list']
    pap_predvol =  dictlist[PaP.__name__]['pap_PredVol_list']
    # pap_lnSE_list_df = pd.DataFrame(np.array([pap_lnSE_list[0]]), index=["pap_lnSE"]).transpose()
    # pap_PredVol_list_df = pd.DataFrame(np.array([pap_PredVol_list[0]]), index=["pap_PredVol"]).transpose()
    # pap_lnSE_list_df.to_csv(str(name ) +" pap_lnSE.csv")
    # pap_PredVol_list_df.to_csv(str(name ) +" pap_PredVol.csv")

    # pap_lnSE_list_df = pd.DataFrame(np.array([paplnse]), index=["pap_lnSE"]).transpose()
    pap_lnSE_list_df = paplnse[0]
    # pap_PredVol_list_df = pd.DataFrame(np.array([pap_predvol]), index=["pap_PredVol"]).transpose()
    pap_PredVol_list_df = pap_predvol[0]
    pap_lnSE_list_df.to_csv(str(name ) +" pap_lnSE.csv")
    pap_PredVol_list_df.to_csv(str(name ) +" pap_PredVol.csv")

    # return pap_mse_list, pap_ql_list, pap_lnSE_list, pap_PredVol_list
    return dictlist



def LR(train_set, test_set, warmup_period, name, n_seq, dictlist, filenames=None):

    makedirs('Ridge//Results', 'LinearRegression', name=name)

    if LR.__name__ not in dictlist:

        dictlist[LR.__name__] = dict()

        dictlist[LR.__name__]['lr_optimal_n_list'] = []
        dictlist[LR.__name__]['lr_mse_list'] = []
        dictlist[LR.__name__]['lr_ql_list'] = []
        dictlist[LR.__name__]['lr_lnSE_list'] = []
        dictlist[LR.__name__]['lr_PredVol_list'] = []

    """
            Linear Regression
    """
    print(str('-') * 28 + "\n\nPerforming Linear Regression\n\n")
    # LR model for 10 regressors on the training set
    print("Training ... \n")
    lr_mse_train_list = []
    for n in n_seq:
        MSE = lr.lin_reg_comb(train_set, n, warmup_period, name=name)[0]
        lr_mse_train_list.append(MSE)

        print( "LR MSE for n=" + str(n ) + " is: " + str(MSE))
    lrdf = pd.concat(lr_mse_train_list).set_index(n_seq)
    lrdf.to_csv(str(name ) + " lr_mse_train.csv")
    # n = lr_mse_train_list.index(min(lr_mse_train_list)) + 1  # add one because index starts at zero
    n = lrdf.SumMSE.idxmin() # add one because index starts at zero

    # lr_optimal_n_list.append(n)
    dictlist['LR']['lr_optimal_n_list'].append(n)

    print( "The smallest n for LR is n=" +str(n))
    lrdf.columns = filenames + ['SumMSE']
    ax_LR = lrdf.plot(logy=True, secondary_y='SumMSE',
                      title='Combined MSE vs n (warmup: ' + str(warmup_period) + ')\noptimal n=' + str(n),
                      figsize=(12, 8))
    ax_LR.set(xlabel='Number of Regressors (n)', ylabel='MSE')
    ax_LR.plot(np.nan, 'xkcd:grey', label='SumMSE (right)')
    ax_LR.legend(bbox_to_anchor=(0., -0.02, 1.05, -0.07),
                 ncol=10, borderaxespad=0.)

    plt.savefig(name.replace(".csv", "") + 'Combined LinearReg MSE vs n_warmup_' + str(warmup_period ) +'.png')
    plt.close()
    """--------Testing--------"""

    print('\nTesting ...\n')
    # LR test set. Use the entire training set as the fit for the test set. See code in LR.
    MSE_LR_test, QL_LR_test, ln_SE_LR_test, PredVol_LR_test, b_LR_test, c_LR_test = lr.lin_reg_comb(train_set, n, warmup_period=warmup_period
                                                                                               ,name=name, test=(True, test_set))
    print("LR(" + str(n) + ") test MSE: " + str(MSE_LR_test) + "; test QL: " + str(QL_LR_test))

    dictlist['LR']['lr_mse_list'].append(MSE_LR_test)
    dictlist['LR']['lr_ql_list'].append(QL_LR_test)
    dictlist['LR']['lr_lnSE_list'].append(ln_SE_LR_test)
    dictlist['LR']['lr_PredVol_list'].append(PredVol_LR_test)

    lr_lnse = dictlist['LR']['lr_lnSE_list']
    lr_predvol = dictlist['LR']['lr_PredVol_list']


    lr_lnSE_list_df = lr_lnse[0]
    lr_PredVol_list_df = lr_predvol[0]



    lr_lnSE_list_df.to_csv(str(name ) +"lr_lnSE.csv")
    lr_PredVol_list_df.to_csv(str(name ) +"lr_PredVol.csv")

    return dictlist


"""
Ridge regression. calls RidgeRegression.py and produces plots and outputs.

only does RR1 combined
"""



def RR(train_set, test_set, warmup_period,name, n_seq, lamda_seq, lr_optimal_n_list_benchmark, dictlist,filenames=None):

    makedirs('Ridge//Results', 'RidgeRegression', name=name)

    if RR.__name__ not in dictlist:

        dictlist[RR.__name__] = dict()

        dictlist[RR.__name__]['rr1_mse_list'] = []
        dictlist[RR.__name__]['rr1_ql_list'] = []
        dictlist[RR.__name__]['rr1_lnSE_list'] = []
        dictlist[RR.__name__]['rr1_PredVol_list'] = []
        dictlist[RR.__name__]['rr1_optimal_log_lambda_list'] = []

        dictlist[RR.__name__]['rr2_mse_list'] = []
        dictlist[RR.__name__]['rr2_ql_list'] = []
        dictlist[RR.__name__]['rr2_lnSE_list'] = []
        dictlist[RR.__name__]['rr2_PredVol_list'] = []
        dictlist[RR.__name__]['rr2_optimal_n_list'] = []
        dictlist[RR.__name__]['rr2_optimal_log_lambda_list'] = []


    """
               Ridge Regression
    """

    print(str('-') * 27 + "\n\nPerforming Ridge Regression\n\n")
    print("Training ... \n")

    # Current status: Working code for train set

    dictlist[RR.__name__]['rr1_mse_list_all'] = []
    dictlist[RR.__name__]['rr1_mse_list_train'] = []
    dictlist[RR.__name__]['rr2_mse_list_all'] = []
    dictlist[RR.__name__]['rr2_mse_list_train'] = []

    for lamda in lamda_seq:
        MSE_RR1, QL_RR1, ln_SE_RR1, b_RR1, c_RR1 = rr.ridge_reg(train_set, lr_optimal_n_list_benchmark,
                                                                warmup_period, name=name, lamda=lamda)
        # rr1_mse_list_all.append(MSE_RR1)
        dictlist[RR.__name__]['rr1_mse_list_all'].append(MSE_RR1)

        print("RR1 MSE for n=" + str(lr_optimal_n_list_benchmark) + ' and lamda=' + str(lamda) + " is: " + str(
            MSE_RR1))

    # for n in n_seq:
    #     for lamda in lamda_seq:
    #         MSE_RR2, QL_RR2, ln_SE_RR2, b_RR2, c_RR2 = rr.ridge_reg(train_set, n, warmup_period, name=name, lamda=lamda)
    #         # rr2_mse_list_all.append(MSE_RR2)
    #         dictlist[RR.__name__]['rr2_mse_list_all'].append(MSE_RR2)
    #
    #         print("RR2 MSE for n=" + str(n) + ' and lamda=' + str(lamda) + " is: " + str(MSE_RR2))

    arrays_RR1 = [np.array(['MSE', 'log_lamda'])]

    rrdf_RR1 = pd.concat(dictlist[RR.__name__]['rr1_mse_list_all'])
    rrdf_RR1['log_lamda'] = np.log(lamda_seq)
    rrdf_RR1.reset_index(inplace=True)

    # min_index_RR1 = dictlist[RR.__name__]['rr1_mse_list_all'].index(min(dictlist[RR.__name__]['rr1_mse_list_all']))
    rr1_optimal_log_lambda = rrdf_RR1.log_lamda[rrdf_RR1.SumMSE.idxmin()]

    # dictlist[RR.__name__]['rr1_mse_list_train'].append(min(dictlist[RR.__name__]['rr1_mse_list_train']))
    dictlist[RR.__name__]['rr1_optimal_log_lambda_list'].append(rr1_optimal_log_lambda)

    # arrays = [np.array(['MSE', 'log_lamda', 'n'])]
    # rrdf = pd.DataFrame(
    #     [np.array(dictlist[RR.__name__]['rr2_mse_list_all']), np.array(np.log(lamda_seq).tolist() * len(n_seq)), np.repeat(n_seq, len(lamda_seq))],
    #     index=arrays).T
    # min_index = dictlist[RR.__name__]['rr2_mse_list_all'].index(min(dictlist[RR.__name__]['rr2_mse_list_all']))
    #
    # rr2_optimal_log_lambda = rrdf.log_lamda[min_index]
    # rr2_optimal_n = rrdf.n[min_index]
    #
    #
    # dictlist[RR.__name__]['rr2_mse_list_train'].append(min( dictlist[RR.__name__]['rr2_mse_list_all']))
    # dictlist[RR.__name__]['rr2_optimal_n_list'].append(rr2_optimal_n)
    # dictlist[RR.__name__]['rr2_optimal_log_lambda_list'].append(rr2_optimal_log_lambda)

    # splits out rr_mse_list into groups of 19, which is the length of the lamda array
    # asdf = [dictlist[RR.__name__]['rr2_mse_list_all'][i:i + len(lamda_seq)] for i in range(0, len(dictlist[RR.__name__]['rr2_mse_list_all']), len(lamda_seq))]
    # blah = []
    # minlamda = []
    # for n in n_seq:
    #     arrays = [np.array(['n=' + str(n), 'n=' + str(n)]), np.array(['MSE', 'log_lambda'])]
    #     arrays = [np.array(['MSE', 'log_lambda'])]
    #     blah.append(pd.DataFrame([asdf[n - 1], np.log(lamda_seq).tolist()], index=arrays).T)
    #     blah[n - 1].to_csv(str(name) + " MSE vs log lambda_n_train" + str(n) + ".csv")
    #     # make a plot of MSE vs lamda for a specific n
    #     blah[n - 1].plot(x='log_lambda', y='MSE', title=str(name) + ' Ridge Regression MSE vs log lambda for n=' + str(n),
    #                      figsize=(9, 6)) \
    #         .legend(loc="center left", bbox_to_anchor=(1, 0.5))
    #     plt.savefig(str(name) + ' Ridge Regression MSE vs log_lambda for n=' + str(n) + '.png')
    #

    """
    Testing
    """

    print('\nTesting ...\n')
    # RR test set. Use the entire training set as the fit for the test set. See code in RR.
    MSE_RR1_test, QL_RR1_test, ln_SE_RR1_test, PredVol_RR1_test, b_RR1_test, c_RR1_test = rr.ridge_reg(train_set, int(
        lr_optimal_n_list_benchmark), warmup_period=warmup_period, lamda=np.exp(rr1_optimal_log_lambda), name=name,
                                                                                                       test=(
                                                                                                       True, test_set))
    print(str(name) + " RR1(" + str(lr_optimal_n_list_benchmark) + ")" + "log_lamdba_" + str(
        rr1_optimal_log_lambda) + " test MSE: " + str(MSE_RR1_test) + "; test QL: " + str(QL_RR1_test))

    # MSE_RR2_test, QL_RR2_test, ln_SE_RR2_test, PredVol_RR2_test, b_RR2_test, c_RR2_test = rr.ridge_reg(train_set,
    #                          int(rr2_optimal_n), warmup_period=warmup_period, lamda=np.exp(rr2_optimal_log_lambda), name=name, test=(True, test_set))
    #
    # print(str(name) + " RR2(" + str(rr2_optimal_n) + ")" + "log_lamdba_" + str(rr2_optimal_log_lambda) + " test MSE: " + str(
    #         MSE_RR2_test) + "; test QL: " + str(QL_RR2_test))


    dictlist[RR.__name__]['rr1_mse_list'].append(MSE_RR1_test)
    dictlist[RR.__name__]['rr1_ql_list'].append(QL_RR1_test)
    dictlist[RR.__name__]['rr1_lnSE_list'].append(ln_SE_RR1_test)
    dictlist[RR.__name__]['rr1_PredVol_list'].append(PredVol_RR1_test)

    rr1_lnse = dictlist[RR.__name__]['rr1_lnSE_list']
    rr1_predvol = dictlist[RR.__name__]['rr1_PredVol_list']


    rr1_lnSE_list_df = rr1_lnse[0]
    rr1_PredVol_list_df = rr1_predvol[0]
    rr1_lnSE_list_df.to_csv(str(name)+" rr1_lnSE.csv")
    rr1_PredVol_list_df.to_csv(str(name)+" rr1_PredVol.csv")
    #
    # dictlist[RR.__name__]['rr2_mse_list'].append(MSE_RR2_test)
    # dictlist[RR.__name__]['rr2_ql_list'].append(QL_RR2_test)
    # dictlist[RR.__name__]['rr2_ql_list'] = []
    # dictlist[RR.__name__]['rr2_lnSE_list'].append(ln_SE_RR2_test)
    # dictlist[RR.__name__]['rr2_PredVol_list'].append(PredVol_RR2_test)
    #
    # rr2_lnse = dictlist[RR.__name__]['rr2_lnSE_list']
    # rr2_predvol = dictlist[RR.__name__]['rr2_PredVol_list']
    #
    # rr2_lnSE_list_df = rr2_lnse[0]
    # rr2_PredVol_list_df = rr2_predvol[0]
    # rr2_lnSE_list_df.to_csv(str(name)+" rr2_lnSE.csv")
    # rr2_PredVol_list_df.to_csv(str(name)+" rr2_PredVol.csv")

    return dictlist

"""
only do BRR1
"""

def BRR(train_set, test_set, warmup_period, name, n_seq, dictlist, param_range, filenames):
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

    makedirs('Ridge//Results', 'BayesianRidgeRegression', name=name)

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
    val_idx = mselists.index(min(mselists))

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
    MSE_BRR1_test, QL_BRR1_test, ln_SE_BRR1_test, PredVol_BRR1_test= brr.bayes_ridge_reg(
        train_set, n, warmup_period ,alpha_1=opt_alpha1, alpha_2=opt_alpha2, lambda_1=opt_lamda1, lambda_2=opt_lamda2,
        test=(True, test_set))

    dictlist[BRR.__name__]['brr1_mse_list'].append(MSE_BRR1_test)
    dictlist[BRR.__name__]['brr1_ql_list'].append(QL_BRR1_test)
    dictlist[BRR.__name__]['brr1_lnSE_list'].append(ln_SE_BRR1_test)
    dictlist[BRR.__name__]['brr1_PredVol_list'].append(PredVol_BRR1_test)


    print(str(name) + " BRR1(" + str(n) + ")" + " test MSE: " + str(MSE_BRR1_test) + "; test QL: " + str(QL_BRR1_test))

    brr1_lnse = dictlist[BRR.__name__]['brr1_lnSE_list']
    brr1_predvol = dictlist[BRR.__name__]['brr1_PredVol_list']

    brr1_lnSE_list_df = brr1_lnse[0]
    brr1_PredVol_list_df = brr1_predvol[0]
    brr1_lnSE_list_df.to_csv(str(name)+" brr1_lnSE.csv")
    brr1_PredVol_list_df.to_csv(str(name)+" brr1_PredVol.csv")

    return dictlist


def KRR(train_set, test_set, warmup_period, name, n_seq, kernels, dictlist, param_range, filenames):
    """
    BEWARE, THE DIFFERENCE IN DICTIONARY NAMING IS DIFFERENT FOR THIS FUNCTION.
    DICTIONARY CONVENTION FOR KRR: 
    dictlist['KRR']['kernel_name']['attrib']

    for ex.
    dictlist['KRR']['gaussian']['krr1_lnSE_list']

    :param train_set: 
    :param test_set: 
    :param warmup_period: 
    :param name: name of currency pair
    :param n_seq: sequence of n vals. for krr1, this is a constant
    :param dictlist: dictionary list of outputs
    :param param_range: tuple where alpha = param_range[0], coef0 = param_range[1], degree = param_range[2]
    :return: 
    """
    makedirs('Ridge//Results', 'KernelRidgeRegression', name=name)
    # kernels var below is for debugging...
    # kernels = ['linear', 'polynomial', 'sigmoid', 'rbf', 'laplacian' ]  #  chi2
    # kernels = ['linear', 'polynomial', 'rbf', 'laplacian']
    # kernels = ['linear', 'polynomial']

    if KRR.__name__ not in dictlist:

        dictlist[KRR.__name__] = dict()
        for kernel_name in kernels:
            # add initial kernel to dict, then add lists to the kernel dictionary
            dictlist[KRR.__name__][kernel_name] = dict()
            dictlist[KRR.__name__][kernel_name]['krr1_mse_list'] = []
            dictlist[KRR.__name__][kernel_name]['krr1_ql_list'] = []
            dictlist[KRR.__name__][kernel_name]['krr1_lnSE_list'] = []
            dictlist[KRR.__name__][kernel_name]['krr1_PredVol_list'] = []

            dictlist[KRR.__name__][kernel_name]['krr_optimal_log_alpha_list'] = []

            if kernel_name is 'polynomial' or kernel_name is 'sigmoid':
                dictlist[KRR.__name__][kernel_name]['krr_optimal_log_coef0_list'] = []
                # only hit this statement if kernel = 'polynomial'
                if kernel_name is 'polynomial':
                    dictlist[KRR.__name__][kernel_name]['krr_optimal_degree_list'] = []

                    #
    """
           Kernel Ridge Regression
    """

    print(str('-') * 34 + "\n\nPerforming Kernel Ridge Regression\n\n")
    # print("Training ... \n")
    # Current status: Working code for train set
    # TODO vary alphas and lamdas while holding n = 9
    # TODO vary a whole bunch of stuff
    # kernels = ['linear','gaussian', 'polynomial', 'sigmoid', 'rbf', 'laplacian' ]  #  chi2

    # KRR1
    n = n_seq
    # looks like i dont need kernel list, but leaving here in case i need it in the future...
    kernellist, mselist, alphalist, coeflist, degreelist = [], [], [], [], []

    for kernel in kernels:
        print("\nTraining ... \n")
        print("Training "+ str(kernel)+" kernel...\n")
        for alpha in param_range[0]:

            if kernel is 'sigmoid':
                for coef0 in param_range[1]:
                    # don't need degree
                    MSE, QL, ln_SE = krr.kernel_ridge_reg(train_set, n, warmup_period, alpha=alpha, coef0=coef0,
                                                          kernel=kernel, test=False)
                    # append results
                    mselist.append(MSE)
                    # kernellist.append(kernel)
                    alphalist.append(alpha)
                    coeflist.append(coef0)
                    # output to console
                    print("KRR for n=" + str(n) + " kernel=" + kernel + " MSE is: " + str(MSE) + " QL : " + str(QL) +
                          " alpha= " + str(alpha) + " coef0= " + str(coef0))

            if kernel is 'polynomial':
                for coef0 in param_range[1]:
                    for degree in param_range[2]:
                        MSE, QL, ln_SE = krr.kernel_ridge_reg(train_set, n, warmup_period, alpha=alpha, coef0=coef0,
                                                              degree=degree, kernel=kernel, test=False)
                        # apppend results
                        mselist.append(MSE)
                        # kernellist.append(kernel)
                        alphalist.append(alpha)
                        coeflist.append(coef0)
                        degreelist.append(degree)
                        # output to console
                        print(
                            "KRR for n=" + str(n) + " kernel=" + kernel + " MSE is: " + str(MSE) + " QL : " + str(QL) +
                            " alpha= " + str(alpha) + " coef0= " + str(coef0) + " degree= " + str(degree))

            else:  # for all other kernels--train
                MSE, QL, ln_SE = krr.kernel_ridge_reg(train_set, n, warmup_period, alpha=alpha, kernel=kernel,
                                                      test=False)
                # apppend results
                mselist.append(MSE)
                # kernellist.append(kernel)
                alphalist.append(alpha)
                # output to console
                print("KRR for n=" + str(n) + " kernel=" + kernel + " MSE is: " + str(MSE) + " QL : " + str(QL) +
                      " alpha= " + str(alpha))

        # at this point in the code, a specific kernel has finished its grid search. now we find the min mse values and
        # save to the dictionary and reset the lists for the next iteration
        mse_df = pd.concat(mselist)
        mse_df.reset_index(drop=True)
        val_idx = mse_df.SumMSE.idxmin()  # add one because index starts at zero

        # for all kernels
        opt_alpha = alphalist[val_idx]
        dictlist[KRR.__name__][kernel]['krr_optimal_log_alpha_list'].append(opt_alpha)

        # for polynomial and sigmoid kernel
        if kernel is 'polynomial' or kernel is 'sigmoid':
            # add optimal coefficient
            opt_coef0 = coeflist[val_idx]
            dictlist[KRR.__name__][kernel]['krr_optimal_log_coef0_list'].append(opt_coef0)

            # for polynomial kernel add stuff to dictionary and test optimal parameters on test set
            if kernel is 'polynomial':
                # add optimal polynomial degree
                opt_deg = degreelist[val_idx]
                dictlist[KRR.__name__][kernel]['krr_optimal_degree_list'].append(opt_deg)

                # test set
                print('\nTesting ...\n')
                # test polynomial
                KRR1_MSE_test, KRR1_QL_test, KRR1_ln_SE_test, KRR1_PredVol = krr.kernel_ridge_reg(train_set, n,
                                                                                                  warmup_period,
                                                                                                  alpha=opt_alpha,
                                                                                                  coef0=opt_coef0,
                                                                                                  degree=opt_deg,
                                                                                                  kernel=kernel,
                                                                                                  test=(True, test_set))

                #   save and output results for polynomial
                dictlist = KRR_append_and_csv(dictlist=dictlist, filename=name, methodname=KRR.__name__,
                                              n=n_seq, kernel=kernel, MSE=KRR1_MSE_test, QL=KRR1_QL_test,
                                              lnSE=KRR1_ln_SE_test, PredVol=KRR1_PredVol)

                print(str(name) + " KRR1(" + str(n) + ")" + " test MSE: " + str(KRR1_MSE_test) + "; test QL: " +
                      str(KRR1_QL_test) + " with kernel= " + str(kernel) + " alpha= " + str(opt_alpha) + " coef0= " +
                      str(opt_coef0) + " degree= " + str(opt_deg))

            else:  # kernel = sigmoid
                # test sigmoid kernel
                print('\nTesting ...\n')
                KRR1_MSE_test, KRR1_QL_test, KRR1_ln_SE_test, KRR1_PredVol = krr.kernel_ridge_reg(train_set, n,
                                                                                                  warmup_period,
                                                                                                  alpha=opt_alpha,
                                                                                                  coef0=opt_coef0,
                                                                                                  kernel=kernel,
                                                                                                  test=(True, test_set))

                #   save and output results for sigmoidal
                dictlist = KRR_append_and_csv(dictlist=dictlist, filename=name, methodname=KRR.__name__,
                                              n=n_seq, kernel=kernel, MSE=KRR1_MSE_test, QL=KRR1_QL_test,
                                              lnSE=KRR1_ln_SE_test, PredVol=KRR1_PredVol)
                print(str(name) + " KRR1(" + str(n) + ")" + " test MSE: " + str(KRR1_MSE_test) + "; test QL: " +
                      str(KRR1_QL_test) + " with kernel= " + str(kernel) + " alpha= " + str(opt_alpha) + " coef0= " +
                      str(opt_coef0))

        else:  # for all other kernels
            # test kernel
            print('\nTesting ...\n')
            KRR1_MSE_test, KRR1_QL_test, KRR1_ln_SE_test, KRR1_PredVol = krr.kernel_ridge_reg(train_set, n,
                                                                                              warmup_period,
                                                                                              alpha=opt_alpha,
                                                                                              kernel=kernel,
                                                                                              test=(True, test_set))

            #   save and output results for all other kernels
            dictlist = KRR_append_and_csv(dictlist=dictlist, filename=name, methodname=KRR.__name__,
                                          n=n_seq, kernel=kernel, MSE=KRR1_MSE_test, QL=KRR1_QL_test,
                                          lnSE=KRR1_ln_SE_test, PredVol=KRR1_PredVol)
            print(str(name) + " KRR1(" + str(n) + ")" + " test MSE: " + str(KRR1_MSE_test) + "; test QL: " +
                  str(KRR1_QL_test) + " with kernel= " + str(kernel) + " alpha= " + str(opt_alpha))

        # reset lists
        kernellist, mselist, alphalist, coeflist, degreelist = [], [], [], [], []

        # the way the code is structured here is that once the kernel finishes training it immediately goes into testing
        # i do not test all the models once all the models are trained.

    # feel free to put a breakpoint in the line below...
    # print('hi')
    print('\n')
    return dictlist, kernels
