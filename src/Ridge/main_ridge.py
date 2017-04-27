import pandas as pd
import linear_regression as lr
import read_in_files as rd
import Volatility as vol
import split_data as sd
from PastAsPresent import PastAsPresent as pap
import RidgeRegression as rr
import BayesianRegression as brr
import KernelRidgeRegression as krr
import matplotlib.pyplot as plt
import numpy as np


dailyvol_zeroes= pd.DataFrame()
# filenames = ['NZDUSD.csv']
filenames = ['AUDUSD.csv', 'CADUSD.csv',  'CHFUSD.csv', 'EURUSD.csv', 'GBPUSD.csv', 'NOKUSD.csv', 'NZDUSD.csv']
filenames_nocsv = [name.replace(".csv", "") for name in filenames]

n_series = np.arange(1,16,1)
# for ridge regression 1 & ridge regression 2
alpha_series = np.exp( np.arange(-6.5, 2.6, 0.5))


# vars
warmup_period = 300

pap_mse_list = []
pap_ql_list = []
pap_lnSE_list = []
pap_PredVol_list = []

lr_optimal_n_list = []
lr_mse_list = []
lr_ql_list = []
lr_lnSE_list = []
lr_PredVol_list = []

rr1_mse_list = []
rr1_ql_list = []
rr1_lnSE_list = []
rr1_PredVol_list = []

rr2_mse_list = []
rr2_ql_list = []
rr2_lnSE_list = []
rr2_PredVol_list = []

brr1_mse_list = []
brr1_ql_list = []
brr1_lnSE_list = []
brr1_PredVol_list = []

brr2_mse_list = []
brr2_ql_list = []
brr2_lnSE_list = []
brr2_PredVol_list = []

krr_linear_mse_list = []
krr_linear_ql_list = []
krr_linear_lnSE_list = []
krr_linear_PredVol_list = []

krr_Gaussian_mse_list = []
krr_Gaussian_ql_list = []
krr_Gaussian_lnSE_list = []
krr_Gaussiany_PredVol_list = []

krr_rbf_mse_list = []
krr_rbf_ql_list = []
krr_rbf_lnSE_list = []
krr_rbf_PredVol_list = []

krr_chi2_mse_list = []
krr_chi2_ql_list = []
krr_chi2_lnSE_list = []
krr_chi2_PredVol_list = []

krr_sigmoid_mse_list = []
krr_sigmoid_ql_list = []
krr_sigmoid_lnSE_list = []
krr_sigmoid_PredVol_list = []

krr_poly_mse_list = []
krr_poly_ql_list = []
krr_poly_lnSE_list = []
krr_poly_PredVol_list = []

for count, name in enumerate(filenames):
    # initialize some lists

    print("Running file: " + str(name))
    #  reads in the files and puts them into dataframes, returns a dataframe called df
    df, df_single_day = rd.read_in_files(name, day=1)
    name = name.split('.')[0]
    # name.append(name)


    # daily_vol_result is the entire vol dataset
    daily_vol_result, daily_ret, daily_vol_zeroes, daily_ret_zeroes = vol.time_vol_calc(df_single_day)

    #  Split the dataset into train and test set
    #  909 is break point for train/test
    train_set, test_set = sd.split_data(dataframe=daily_vol_result, idx=910, reset_index=False)

    # """
    #         PastAsPresent
    # """
    # print(str('-') * 24 + "\n\nPerforming PastAsPresent\n\n")
    #
    # # PastAsPresent -- Test sample only
    # papMSE_test, papQL_test, pap_ln_SE_test,pap_PredVol_test = pap.tn_pred_tn_plus_1(test_set)
    # print("Past as Present MSE: " + str(papMSE_test) + "; QL: " + str(papQL_test))
    #
    # pap_mse_list.append(papMSE_test)
    # pap_ql_list.append(papQL_test)
    # pap_lnSE_list.append(pap_ln_SE_test)
    # pap_PredVol_list.append(pap_PredVol_test)

    """
            Linear Regression
    """
    print(str('-') * 28 + "\n\nPerforming Linear Regression\n\n")
    # LR model for 10 regressors on the training set
    print("Training ... \n")
    lr_mse_train_list = []
    for n in range(1,16):
        MSE = lr.lin_reg(train_set, n, warmup_period)[0]
        lr_mse_train_list.append(MSE)

        print("LR MSE for n="+str(n)+" is: "+str(MSE))

    n = lr_mse_train_list.index(min(lr_mse_train_list)) + 1  # add one because index starts at zero
    lr_optimal_n_list.append(n)
    print("The smallest n for LR is n="+str(n))
    figLR = plt.figure(figsize=(8, 6))
    ax_LR = figLR.add_subplot(111)
    ax_LR.plot(range(1, 16), lr_mse_train_list)
    ax_LR.set(title=name.replace(".csv","")+' MSE vs n (warmup: '+str(warmup_period)+')\noptimal n='+str(n), xlabel='number of regressors', ylabel='MSE')
    plt.savefig(name.replace(".csv","")+' LinearReg MSE vs n_warmup_'+str(warmup_period)+'.png')

    print('\nTesting ...\n')
    # LR test set. Use the entire training set as the fit for the test set. See code in LR.
    MSE_LR_test, QL_LR_test, ln_SE_LR_test, PredVol_LR_test, b_LR_test, c_LR_test = lr.lin_reg(train_set, n, warmup_period=warmup_period, test=(True, test_set))
    print("LR("+str(n)+") test MSE: "+str(MSE_LR_test)+"; test QL: "+str(QL_LR_test))

    lr_mse_list.append(MSE_LR_test)
    lr_ql_list.append(QL_LR_test)
    lr_lnSE_list.append(ln_SE_LR_test)
    lr_PredVol_list.append(PredVol_LR_test)



    # """
    #         Ridge Regression
    # """
    #
    # print(str('-') * 27 + "\n\nPerforming Ridge Regression\n\n")
    # print("Training ... \n")
    #
    # # Current status: Working code for train set
    # # TODO vary lamda while holding n = 9
    # # TODO vary both n and lamda
    # for n in range(1,11):
    #     MSE, QL, ln_SE, b, c = rr.ridge_reg(train_set, n, warmup_period,lamda=1)
    #     rr_mse_list.append(MSE)
    #
    #     print("RR MSE for n="+str(n)+" is: "+str(MSE))
    #
    # n = rr_mse_list.index(min(rr_mse_list)) + 1  # add one because index starts at zero
    # print("The smallest n for RR is n="+str(n))
    # print('\nTesting ...\n')
    #
    # """
    #        Bayesian Ridge Regression
    # """
    # print(str('-') * 36 + "\n\nPerforming Bayesian Ridge Regression\n\n")
    # print("Training ... \n")
    # # Current status: Working code for train set
    # # TODO vary alphas and lamdas while holding n = 9
    # # TODO vary both n and lamdas and alphas
    # for n in range(1,11):
    #     MSE, QL, ln_SE, b, c = brr.bayes_ridge_reg(train_set, n, warmup_period, alpha_1=1e-06, alpha_2=1e-06,
    #                                                lambda_1=1e-06, lambda_2=1e-06, test=False)
    #     brr_mse_list.append(MSE)
    #
    #     print("BRR MSE for n="+str(n)+" is: "+str(MSE))
    #
    # n = brr_mse_list.index(min(brr_mse_list)) + 1  # add one because index starts at zero
    # print("The smallest n for BRR is n="+str(n))
    # print('\nTesting ...\n')
    #
    # """
    #        Kernel Ridge Regression
    # """
    #
    # print(str('-') * 34 + "\n\nPerforming Kernel Ridge Regression\n\n")
    # print("Training ... \n")
    # # Current status: Working code for train set
    # # TODO vary alphas and lamdas while holding n = 9
    # # TODO vary a whole bunch of stuff
    # kernels=['linear', 'polynomial', 'sigmoid', 'rbf', 'laplacian' ]  #  chi2
    #
    # for kernel in kernels:
    #     for n in range(1,11):
    #         MSE, QL, ln_SE = krr.kernel_ridge_reg(train_set, n, warmup_period, alpha=1,coef0=1, degree=3, kernel=kernel, test=False)
    #         krr_mse_list.append(MSE)
    #         print("KRR MSE for n=" + str(n) + " and kernel=" + kernel + " is: " + str(MSE))
    #
    #     n = krr_mse_list.index(min(krr_mse_list)) + 1  # add one because index starts at zero
    #     print("The smallest n for KRR is n=" + str(n) + " for kernel = " + kernel)
    #     print("\nTraining ... \n")
    #
    # print('\nTesting ...\n')

    # feel free to put a breakpoint in the line below...
    print('hi')

# pap_test_restuls_df = pd.DataFrame({"PastAsPresent MSE":pap_mse_list,
#                                      "PastAsPresent QL": pap_ql_list})
# pap_test_restuls_df = pap_test_restuls_df.set_index(np.transpose(filenames_nocsv), drop=True)
# pap_test_restuls_df.to_csv('PastAsPresent_test_MSE_QL.csv')


lr_test_restuls_df = pd.DataFrame({"LinearReg MSE":lr_mse_list,
                                   "LinearReg QL": lr_ql_list,
                                   "Optimal n": lr_optimal_n_list})
lr_test_restuls_df = lr_test_restuls_df.set_index(np.transpose(filenames_nocsv), drop=True)
lr_test_restuls_df.to_csv('LinearReg_test_MSE_QL_warmup_'+str(warmup_period)+'.csv')
print('hi')
