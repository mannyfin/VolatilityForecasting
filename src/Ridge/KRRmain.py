
import pandas as pd
import numpy as np
import KernelRidgeRegression as krr
from makedirs import makedirs
from KRR_append_and_csv import KRR_append_and_csv
import os

def KRR(train_set, test_set, warmup_period, name,count, n_seq, kernels, dictlist, param_range):
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
    :param count: counter
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
                          " alpha= "+ str(alpha) + " coef0= " + str(coef0))

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
                        print("KRR for n=" + str(n) + " kernel=" + kernel + " MSE is: " + str(MSE) + " QL : " + str(QL) +
                              " alpha= " + str(alpha) + " coef0= " + str(coef0) + " degree= " + str(degree))

            else: #for all other kernels--train
                MSE, QL, ln_SE = krr.kernel_ridge_reg(train_set, n, warmup_period, alpha=alpha, kernel=kernel,test=False)
                # apppend results
                mselist.append(MSE)
                # kernellist.append(kernel)
                alphalist.append(alpha)
                # output to console
                print("KRR for n=" + str(n) + " kernel=" + kernel  + " MSE is: " + str(MSE) + " QL : " + str(QL) +
                      " alpha= " + str(alpha))

        # at this point in the code, a specific kernel has finished its grid search. now we find the min mse values and
        # save to the dictionary and reset the lists for the next iteration

        val_idx = mselist.index(min(mselist))  # add one because index starts at zero

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
                dictlist = KRR_append_and_csv(dictlist=dictlist, filename=name, count=count, methodname=KRR.__name__,
                                              n=n_seq, kernel=kernel, MSE=KRR1_MSE_test, QL=KRR1_QL_test,
                                              lnSE=KRR1_ln_SE_test, PredVol=KRR1_PredVol)

                print(str(name) + " KRR1(" + str(n) + ")" + " test MSE: " + str(KRR1_MSE_test) + "; test QL: " +
                      str(KRR1_QL_test) + " with kernel= " + str(kernel)+ " alpha= "+ str(opt_alpha) + " coef0= " +
                      str(opt_coef0) + " degree= " + str(opt_deg))

            else: #kernel = sigmoid
                # test sigmoid kernel
                print('\nTesting ...\n')
                KRR1_MSE_test, KRR1_QL_test, KRR1_ln_SE_test, KRR1_PredVol = krr.kernel_ridge_reg(train_set, n,
                                                                                              warmup_period,
                                                                                              alpha=opt_alpha,
                                                                                              coef0=opt_coef0,
                                                                                              kernel=kernel,
                                                                                              test=(True, test_set))

            #   save and output results for sigmoidal
                dictlist = KRR_append_and_csv(dictlist=dictlist, filename=name, count=count, methodname=KRR.__name__,
                                              n=n_seq, kernel=kernel, MSE=KRR1_MSE_test, QL=KRR1_QL_test,
                                              lnSE=KRR1_ln_SE_test, PredVol=KRR1_PredVol)
                print(str(name) + " KRR1(" + str(n) + ")" + " test MSE: " + str(KRR1_MSE_test) + "; test QL: " +
                      str(KRR1_QL_test) + " with kernel= " + str(kernel)+ " alpha= "+ str(opt_alpha) + " coef0= " +
                      str(opt_coef0))

        else: # for all other kernels
                # test kernel
                print('\nTesting ...\n')
                KRR1_MSE_test, KRR1_QL_test, KRR1_ln_SE_test, KRR1_PredVol = krr.kernel_ridge_reg(train_set, n,
                                                                                              warmup_period,
                                                                                              alpha=opt_alpha,
                                                                                              kernel=kernel,
                                                                                              test=(True, test_set))

            #   save and output results for all other kernels
                dictlist = KRR_append_and_csv(dictlist=dictlist, filename=name, count=count, methodname=KRR.__name__,
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
