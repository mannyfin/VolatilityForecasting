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
# filenames = ['AUDUSD.csv']
filenames = ['AUDUSD.csv', 'CADUSD.csv',  'CHFUSD.csv', 'EURUSD.csv', 'GBPUSD.csv', 'NOKUSD.csv', 'NZDUSD.csv']
filenames_nocsv = [name.replace(".csv", "") for name in filenames]

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
rr1_optimal_log_lambda_list = []

rr2_mse_list = []
rr2_ql_list = []
rr2_lnSE_list = []
rr2_PredVol_list = []
rr2_optimal_n_list = []
rr2_optimal_log_lambda_list = []

brr_mse_list = []
brr_ql_list = []
brr_lnSE_list = []
brr_PredVol_list = []
brr_optimal_n_list = []
brr_optimal_log_alpha1_list = []
brr_optimal_log_alpha2_list = []
brr_optimal_log_lambda1_list = []
brr_optimal_log_lambda2_list = []

krr_linear_mse_list = []
krr_linear_ql_list = []
krr_linear_lnSE_list = []
krr_linear_PredVol_list = []
krr_linear_log_alpha_list = []
krr_linear_gamma_list = []

krr_Gaussian_mse_list = []
krr_Gaussian_ql_list = []
krr_Gaussian_lnSE_list = []
krr_Gaussian_PredVol_list = []
krr_Gaussian_log_alpha_list = []
krr_Gaussian_gamma_list = []

krr_rbf_mse_list = []
krr_rbf_ql_list = []
krr_rbf_lnSE_list = []
krr_rbf_PredVol_list = []
krr_rbf_log_alpha_list = []
krr_rbf_gamma_list = []

krr_chi2_mse_list = []
krr_chi2_ql_list = []
krr_chi2_lnSE_list = []
krr_chi2_PredVol_list = []
krr_chi2_log_alpha_list = []
krr_chi2_gamma_list = []

krr_sigmoid_mse_list = []
krr_sigmoid_ql_list = []
krr_sigmoid_lnSE_list = []
krr_sigmoid_PredVol_list = []
krr_sigmoid_log_alpha_list = []
krr_sigmoid_gamma_list = []

krr_poly_mse_list = []
krr_poly_ql_list = []
krr_poly_lnSE_list = []
krr_poly_PredVol_list = []
krr_poly_log_alpha_list = []
krr_poly_gamma_list = []
krr_poly_degree_list = []

# vars
warmup_period = 300
lr_optimal_n_list_benchmark = [9,7,7,7,7,7,10]
n_seq = np.arange(1, 16, 1)
lamda_seq = np.exp(np.arange(-1, 3.1, 0.2))  # for RR1 and RR2

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

    """
            PastAsPresent
    """
    print(str('-') * 24 + "\n\nPerforming PastAsPresent\n\n")

    # PastAsPresent -- Test sample only
    papMSE_test, papQL_test, pap_ln_SE_test,pap_PredVol_test = pap.tn_pred_tn_plus_1(test_set)
    print("Past as Present MSE: " + str(papMSE_test) + "; QL: " + str(papQL_test))

    pap_mse_list.append(papMSE_test)
    pap_ql_list.append(papQL_test)
    pap_lnSE_list.append(pap_ln_SE_test)
    pap_PredVol_list.append(pap_PredVol_test)

    pap_lnSE_list_df = pd.DataFrame(np.array([pap_lnSE_list[0]]), index=["pap_lnSE"]).transpose()
    pap_PredVol_list_df = pd.DataFrame(np.array([pap_PredVol_list[0]]), index=["pap_PredVol"]).transpose()
    pap_lnSE_list_df.to_csv(str(name)+" pap_lnSE.csv")
    pap_PredVol_list_df.to_csv(str(name)+" pap_PredVol.csv")

    # """
    #         Linear Regression
    # """
    # print(str('-') * 28 + "\n\nPerforming Linear Regression\n\n")
    # # LR model for 10 regressors on the training set
    # print("Training ... \n")
    # lr_mse_train_list = []
    # for n in n_seq:
    #     MSE = lr.lin_reg(train_set, n, warmup_period, name=name)[0]
    #     lr_mse_train_list.append(MSE)
    #
    #     print("LR MSE for n="+str(n)+" is: "+str(MSE))
    # lrdf = pd.DataFrame([n_seq,lr_mse_train_list],index=["n","lr_mse_train"])
    # lrdf.to_csv(str(name)+" lr_mse_train.csv")
    # n = lr_mse_train_list.index(min(lr_mse_train_list)) + 1  # add one because index starts at zero
    # lr_optimal_n_list.append(n)
    # print("The smallest n for LR is n="+str(n))
    # figLR = plt.figure(figsize=(8, 6))
    # ax_LR = figLR.add_subplot(111)
    # ax_LR.plot(range(1, 16), lr_mse_train_list)
    # ax_LR.set(title=name.replace(".csv","")+' MSE vs n (warmup: '+str(warmup_period)+')\noptimal n='+str(n), xlabel='number of regressors', ylabel='MSE')
    # plt.savefig(name.replace(".csv","")+' LinearReg MSE vs n_warmup_'+str(warmup_period)+'.png')
    #
    # print('\nTesting ...\n')
    # # LR test set. Use the entire training set as the fit for the test set. See code in LR.
    # MSE_LR_test, QL_LR_test, ln_SE_LR_test, PredVol_LR_test, b_LR_test, c_LR_test = lr.lin_reg(train_set, n, warmup_period=warmup_period,name=name, test=(True, test_set))
    # print("LR("+str(n)+") test MSE: "+str(MSE_LR_test)+"; test QL: "+str(QL_LR_test))
    #
    # lr_mse_list.append(MSE_LR_test)
    # lr_ql_list.append(QL_LR_test)
    # lr_lnSE_list.append(ln_SE_LR_test)
    # lr_PredVol_list.append(PredVol_LR_test)
    #
    # lr_lnSE_list_df = pd.DataFrame(np.array([lr_lnSE_list[0]]), index=["lr_lnSE"]).transpose()
    # lr_PredVol_list_df = pd.DataFrame(np.array([lr_PredVol_list[0]]), index=["lr_PredVol"]).transpose()
    # lr_lnSE_list_df.to_csv(str(name)+"lr_lnSE.csv")
    # lr_PredVol_list_df.to_csv(str(name)+"lr_PredVol.csv")


    """
            Ridge Regression
    """

    print(str('-') * 27 + "\n\nPerforming Ridge Regression\n\n")
    print("Training ... \n")

    # Current status: Working code for train set

    # n_seq = np.arange(1,3,1)
    # lamda_seq = np.exp(np.arange(-6.5, -5.5, 0.5))
    rr1_mse_list_all = []
    rr1_mse_list_train = []
    rr2_mse_list_all = []
    rr2_mse_list_train = []
    for lamda in lamda_seq:

        MSE_RR1, QL_RR1, ln_SE_RR1, b_RR1, c_RR1 = rr.ridge_reg(train_set, n, warmup_period, name=name, lamda=lamda)
        rr1_mse_list_all.append(MSE_RR1)

        print("RR1 MSE for n="+str(lr_optimal_n_list_benchmark[count]) + ' and lamda='+str(lamda)+" is: "+str(MSE_RR1))

    for n in n_seq:
        for lamda in lamda_seq:

            MSE_RR2, QL_RR2, ln_SE_RR2, b_RR2, c_RR2 = rr.ridge_reg(train_set, n, warmup_period,name=name, lamda=lamda)
            rr2_mse_list_all.append(MSE_RR2)

            print("RR2 MSE for n="+str(n) + ' and lamda='+str(lamda)+" is: "+str(MSE_RR2))

    # n = rr_mse_list_all.index(min(rr_mse_list_all)) + 1  # add one because index starts at zero
    # print("The smallest n for RR is n="+str(n))

    arrays_RR1 = [np.array(['MSE', 'log_lamda'])]
    rrdf_RR1 = pd.DataFrame([np.array(rr1_mse_list_all), np.array(np.log(lamda_seq))], index=arrays_RR1).T
    min_index_RR1 = rr1_mse_list_all.index(min(rr1_mse_list_all))
    rr1_optimal_log_lambda = rrdf_RR1.log_lamda[min_index_RR1]
    # min MSE for a file
    rr1_mse_list_train.append(min(rr1_mse_list_all))
    rr1_optimal_log_lambda_list.append(rr1_optimal_log_lambda)

    arrays = [np.array(['MSE', 'log_lamda', 'n'])]
    rrdf = pd.DataFrame([np.array(rr2_mse_list_all), np.array(np.log(lamda_seq).tolist()*len(n_seq)),np.repeat(n_seq, len(lamda_seq))], index=arrays).T
    min_index = rr2_mse_list_all.index(min(rr2_mse_list_all))
    rr2_optimal_log_lambda = rrdf.log_lamda[min_index]
    rr2_optimal_n = rrdf.n[min_index]
    rr2_mse_list_train.append(min(rr2_mse_list_all))
    rr2_optimal_n_list.append(rr2_optimal_n)
    rr2_optimal_log_lambda_list.append(rr2_optimal_log_lambda)

    # splits out rr_mse_list into groups of 19, which is the length of the lamda array
    asdf = [rr2_mse_list_all[i:i + len(lamda_seq)] for i in range(0, len(rr2_mse_list_all), len(lamda_seq))]
    blah=[]
    minlamda=[]
    for n in n_seq:
        arrays = [np.array(['n=' + str(n), 'n=' + str(n)]), np.array(['MSE', 'log_lambda'])]
        arrays = [np.array(['MSE', 'log_lambda'])]
        blah.append(pd.DataFrame([asdf[n-1], np.log(lamda_seq).tolist()], index=arrays).T)
        blah[n - 1].to_csv(str(name)+" MSE vs log lambda_n_train"+str(n)+".csv")
        # make a plot of MSE vs lamda for a specific n
        blah[n-1].plot(x='log_lambda', y='MSE', title=str(name)+' Ridge Regression MSE vs log lambda for n=' + str(n), figsize=(9, 6)) \
            .legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig(str(name)+' Ridge Regression MSE vs log_lambda for n=' + str(n)+'.png')
        # minlamda.append(blah[n-1]['lamda'][blah[n-1]['MSE'].idxmin()])

    print('\nTesting ...\n')
    # RR test set. Use the entire training set as the fit for the test set. See code in RR.
    MSE_RR1_test, QL_RR1_test, ln_SE_RR1_test, PredVol_RR1_test, b_RR1_test, c_RR1_test = rr.ridge_reg(train_set, int(lr_optimal_n_list_benchmark[count]), warmup_period=warmup_period,lamda = np.exp(rr1_optimal_log_lambda), name=name, test=(True, test_set))
    print(str(name)+" RR1("+str(lr_optimal_n_list_benchmark[count])+")"+"log_lamdba_"+str(rr1_optimal_log_lambda)+" test MSE: "+str(MSE_RR1_test)+"; test QL: "+str(QL_RR1_test))

    MSE_RR2_test, QL_RR2_test, ln_SE_RR2_test, PredVol_RR2_test, b_RR2_test, c_RR2_test = rr.ridge_reg(train_set, int(rr2_optimal_n), warmup_period=warmup_period,lamda = np.exp(rr2_optimal_log_lambda), name=name, test=(True, test_set))
    print(str(name)+" RR2("+str(rr2_optimal_n)+")"+"log_lamdba_"+str(rr2_optimal_log_lambda)+" test MSE: "+str(MSE_RR2_test)+"; test QL: "+str(QL_RR2_test))

    rr1_mse_list.append(MSE_RR1_test)
    rr1_ql_list.append(QL_RR1_test)
    rr1_lnSE_list.append(ln_SE_RR1_test)
    rr1_PredVol_list.append(PredVol_RR1_test)

    rr1_lnSE_list_df = pd.DataFrame(np.array([rr1_lnSE_list[0]]), index=["rr1_lnSE"]).transpose()
    rr1_PredVol_list_df = pd.DataFrame(np.array([rr1_PredVol_list[0]]), index=["rr1_PredVol"]).transpose()
    rr1_lnSE_list_df.to_csv(str(name)+" rr1_lnSE.csv")
    rr1_PredVol_list_df.to_csv(str(name)+" rr1_PredVol.csv")

    rr2_mse_list.append(MSE_RR2_test)
    rr2_ql_list.append(QL_RR2_test)
    rr2_lnSE_list.append(ln_SE_RR2_test)
    rr2_PredVol_list.append(PredVol_RR2_test)

    rr2_lnSE_list_df = pd.DataFrame(np.array([rr2_lnSE_list[0]]), index=["rr2_lnSE"]).transpose()
    rr2_PredVol_list_df = pd.DataFrame(np.array([rr2_PredVol_list[0]]), index=["rr2_PredVol"]).transpose()
    rr2_lnSE_list_df.to_csv(str(name)+" rr2_lnSE.csv")
    rr2_PredVol_list_df.to_csv(str(name)+" rr2_PredVol.csv")


    """
           Bayesian Ridge Regression
    """
    print(str('-') * 36 + "\n\nPerforming Bayesian Ridge Regression\n\n")
    print("Training ... \n")
    # Current status: Working code for train set
    # TODO vary alphas and lamdas while holding n = 9
    # TODO vary both n and lamdas and alphas
    for n in range(1,11):
        for alpha1 in np.exp(np.arange(-17,-3,1)):
            for alpha2 in np.exp(np.arange(-17,-3,1)):
                for lamda1 in np.exp(np.arange(-17,-3,1)):
                    for lamda2 in np.exp(np.arange(-17,-3,1)):

                        MSE, QL, ln_SE, b, c = brr.bayes_ridge_reg(train_set, n, warmup_period, alpha_1=alpha1, alpha_2=alpha2,
                                                                   lambda_1=lamda1, lambda_2=lamda2, test=False)
                        brr_mse_list.append(MSE)

                        print("BRR MSE for n="+str(n)+" is: "+str(MSE))

    n = brr_mse_list.index(min(brr_mse_list)) + 1  # add one because index starts at zero
    print("The smallest n for BRR is n="+str(n))
    print('\nTesting ...\n')




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

pap_test_restuls_df = pd.DataFrame({"PastAsPresent MSE":pap_mse_list,
                                     "PastAsPresent QL": pap_ql_list})
pap_test_restuls_df = pap_test_restuls_df.set_index(np.transpose(filenames_nocsv), drop=True)
pap_test_restuls_df.to_csv('PastAsPresent_test_MSE_QL.csv')


lr_test_restuls_df = pd.DataFrame({"LinearReg MSE":lr_mse_list,
                                   "LinearReg QL": lr_ql_list,
                                   "Optimal n": lr_optimal_n_list})
lr_test_restuls_df = lr_test_restuls_df.set_index(np.transpose(filenames_nocsv), drop=True)
lr_test_restuls_df.to_csv('LinearReg_test_MSE_QL_warmup_'+str(warmup_period)+'.csv')

rr1_test_restuls_df = pd.DataFrame({"Ridge Regression1 MSE":rr1_mse_list,
                                     "Ridge Regression1 QL": rr1_ql_list,
                                   "Optimal log_lambda": rr1_optimal_log_lambda_list})
rr1_test_restuls_df = rr1_test_restuls_df.set_index(np.transpose(filenames_nocsv), drop=True)
rr1_test_restuls_df.to_csv('Ridge_Regression1_test_MSE_QL.csv')

rr2_test_restuls_df = pd.DataFrame({"Ridge Regression2 MSE":rr2_mse_list,
                                     "Ridge Regression2 QL": rr2_ql_list,
                                   "Optimal n": rr2_optimal_n_list,
                                   "Optimal log_lambda": rr2_optimal_log_lambda_list})
rr2_test_restuls_df = rr2_test_restuls_df.set_index(np.transpose(filenames_nocsv), drop=True)
rr2_test_restuls_df.to_csv('Ridge_Regression2_test_MSE_QL.csv')

print('hi')

