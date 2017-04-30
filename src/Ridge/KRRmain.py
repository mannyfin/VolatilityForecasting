
import pandas as pd
import numpy as np
import KernelRidgeRegression as krr
from makedirs import makedirs


def KRR(train_set, test_set, warmup_period, name,count, n_seq, dictlist, param_range):
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
    kernels = ['linear','gaussian', 'polynomial', 'sigmoid', 'rbf', 'laplacian' ]  #  chi2

    if KRR.__name__ not in dictlist:

        dictlist[KRR.__name__] = dict()
        for kernel_name in kernels:
            dictlist[KRR.__name__][kernel_name]['krr1_mse_list'] = []
            dictlist[KRR.__name__][kernel_name]['krr1_ql_list'] = []
            dictlist[KRR.__name__][kernel_name]['krr1_lnSE_list'] = []
            dictlist[KRR.__name__][kernel_name]['krr1_PredVol_list'] = []

            dictlist[KRR.__name__][kernel_name]['krr_optimal_log_alpha_list'] = []

            if kernel_name is 'polynomial' or kernel_name is 'sigmoid':
                dictlist[KRR.__name__][kernel_name]['krr_optimal_log_coef0_list'] = []

            if kernel_name is 'polynomial':
                dictlist[KRR.__name__][kernel_name]['krr_optimal_degree_list'] = []

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

    # KRR1
    n = n_seq

    kernellist, mselist, alphalist, coeflist, degreelist = [], [], [], [], []


    for kernel in kernels:
        for alpha in param_range[0]:

            if kernel is 'sigmoid':
                for coef0 in param_range[1]:
                    # don't need degree
                    MSE, QL, ln_SE = krr.kernel_ridge_reg(train_set, n, warmup_period, alpha=alpha, coef0=coef0,
                                                          kernel=kernel, test=False)
                    mselist.append(MSE)
                    kernellist.append(kernel)
                    alphalist.append(alpha)
                    coeflist.append(coef0)

                    print("KRR for n=" + str(n) + " kernel=" + kernel +" alpha= "+ str(alpha) + "coef0= " + str(coef0)+
                          " MSE is: " + str(MSE) + " QL : " + str(QL))

            if kernel is 'polynomial':
                for coef0 in param_range[1]:
                    for degree in param_range[2]:
                        MSE, QL, ln_SE = krr.kernel_ridge_reg(train_set, n, warmup_period, alpha=alpha, coef0=coef0,
                                                              degree=degree, kernel=kernel, test=False)
                        mselist.append(MSE)
                        kernellist.append(kernel)
                        alphalist.append(alpha)
                        coeflist.append(coef0)
                        degreelist.append(degree)
                        print("KRR for n=" + str(n) + " kernel=" + kernel + " alpha= " + str(alpha) + "coef0= " +
                              str(coef0) + " degree= " + str(degree) + " MSE is: " + str(MSE) + " QL : " + str(QL))

            else:
                MSE, QL, ln_SE = krr.kernel_ridge_reg(train_set, n, warmup_period, alpha=alpha, kernel=kernel,test=False)

                mselist.append(MSE)
                kernellist.append(kernel)
                alphalist.append(alpha)


            krr_mse_list.append(MSE)


        n = krr_mse_list.index(min(krr_mse_list)) + 1  # add one because index starts at zero
        print("The smallest n for KRR is n=" + str(n) + " for kernel = " + kernel)
        print("\nTraining ... \n")

    print('\nTesting ...\n')

    # feel free to put a breakpoint in the line below...
    print('hi')