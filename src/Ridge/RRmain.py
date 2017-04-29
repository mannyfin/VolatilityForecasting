"""
Ridge regression. calls RidgeRegression.py and produces plots and outputs.
"""

import numpy as np
import pandas as pd
import RidgeRegression as rr
import matplotlib.pyplot as plt


def RR(train_set, test_set, warmup_period, name,n_seq, lamda_seq, lr_optimal_n_list_benchmark, count, filenames_nocsv):
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
        MSE_RR1, QL_RR1, ln_SE_RR1, b_RR1, c_RR1 = rr.ridge_reg(train_set, lr_optimal_n_list_benchmark[count],
                                                                warmup_period, name=name, lamda=lamda)
        rr1_mse_list_all.append(MSE_RR1)

        print("RR1 MSE for n=" + str(lr_optimal_n_list_benchmark[count]) + ' and lamda=' + str(lamda) + " is: " + str(
            MSE_RR1))

    for n in n_seq:
        for lamda in lamda_seq:
            MSE_RR2, QL_RR2, ln_SE_RR2, b_RR2, c_RR2 = rr.ridge_reg(train_set, n, warmup_period, name=name, lamda=lamda)
            rr2_mse_list_all.append(MSE_RR2)

            print("RR2 MSE for n=" + str(n) + ' and lamda=' + str(lamda) + " is: " + str(MSE_RR2))

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
    rrdf = pd.DataFrame(
        [np.array(rr2_mse_list_all), np.array(np.log(lamda_seq).tolist() * len(n_seq)), np.repeat(n_seq, len(lamda_seq))],
        index=arrays).T
    min_index = rr2_mse_list_all.index(min(rr2_mse_list_all))
    rr2_optimal_log_lambda = rrdf.log_lamda[min_index]
    rr2_optimal_n = rrdf.n[min_index]
    rr2_mse_list_train.append(min(rr2_mse_list_all))
    rr2_optimal_n_list.append(rr2_optimal_n)
    rr2_optimal_log_lambda_list.append(rr2_optimal_log_lambda)

    # splits out rr_mse_list into groups of 19, which is the length of the lamda array
    asdf = [rr2_mse_list_all[i:i + len(lamda_seq)] for i in range(0, len(rr2_mse_list_all), len(lamda_seq))]
    blah = []
    minlamda = []
    for n in n_seq:
        arrays = [np.array(['n=' + str(n), 'n=' + str(n)]), np.array(['MSE', 'log_lambda'])]
        arrays = [np.array(['MSE', 'log_lambda'])]
        blah.append(pd.DataFrame([asdf[n - 1], np.log(lamda_seq).tolist()], index=arrays).T)
        blah[n - 1].to_csv(str(name) + " MSE vs log lambda_n_train" + str(n) + ".csv")
        # make a plot of MSE vs lamda for a specific n
        blah[n - 1].plot(x='log_lambda', y='MSE', title=str(name) + ' Ridge Regression MSE vs log lambda for n=' + str(n),
                         figsize=(9, 6)) \
            .legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig(str(name) + ' Ridge Regression MSE vs log_lambda for n=' + str(n) + '.png')
        # minlamda.append(blah[n-1]['lamda'][blah[n-1]['MSE'].idxmin()])

    print('\nTesting ...\n')
    # RR test set. Use the entire training set as the fit for the test set. See code in RR.
    MSE_RR1_test, QL_RR1_test, ln_SE_RR1_test, PredVol_RR1_test, b_RR1_test, c_RR1_test = rr.ridge_reg(train_set, int(
        lr_optimal_n_list_benchmark[count]), warmup_period=warmup_period, lamda=np.exp(rr1_optimal_log_lambda), name=name,
                                                                                                       test=(
                                                                                                       True, test_set))
    print(str(name) + " RR1(" + str(lr_optimal_n_list_benchmark[count]) + ")" + "log_lamdba_" + str(
        rr1_optimal_log_lambda) + " test MSE: " + str(MSE_RR1_test) + "; test QL: " + str(QL_RR1_test))

    MSE_RR2_test, QL_RR2_test, ln_SE_RR2_test, PredVol_RR2_test, b_RR2_test, c_RR2_test = rr.ridge_reg(train_set,
                             int(rr2_optimal_n), warmup_period=warmup_period, lamda=np.exp(rr2_optimal_log_lambda), name=name, test=(True, test_set))

    print(str(name) + " RR2(" + str(rr2_optimal_n) + ")" + "log_lamdba_" + str(rr2_optimal_log_lambda) + " test MSE: " + str(
            MSE_RR2_test) + "; test QL: " + str(QL_RR2_test))

    rr1_mse_list.append(MSE_RR1_test)
    rr1_ql_list.append(QL_RR1_test)
    rr1_lnSE_list.append(ln_SE_RR1_test)
    rr1_PredVol_list.append(PredVol_RR1_test)

    rr1_lnSE_list_df = pd.DataFrame(np.array([rr1_lnSE_list[0]]), index=["rr1_lnSE"]).transpose()
    rr1_PredVol_list_df = pd.DataFrame(np.array([rr1_PredVol_list[0]]), index=["rr1_PredVol"]).transpose()
    rr1_lnSE_list_df.to_csv(str(name) + " rr1_lnSE.csv")
    rr1_PredVol_list_df.to_csv(str(name) + " rr1_PredVol.csv")

    rr2_mse_list.append(MSE_RR2_test)
    rr2_ql_list.append(QL_RR2_test)
    rr2_lnSE_list.append(ln_SE_RR2_test)
    rr2_PredVol_list.append(PredVol_RR2_test)

    rr2_lnSE_list_df = pd.DataFrame(np.array([rr2_lnSE_list[0]]), index=["rr2_lnSE"]).transpose()
    rr2_PredVol_list_df = pd.DataFrame(np.array([rr2_PredVol_list[0]]), index=["rr2_PredVol"]).transpose()
    rr2_lnSE_list_df.to_csv(str(name) + " rr2_lnSE.csv")
    rr2_PredVol_list_df.to_csv(str(name) + " rr2_PredVol.csv")

    return rr1_mse_list, rr1_ql_list, rr1_optimal_log_lambda_list
    # rr1_test_restuls_df = pd.DataFrame({"Ridge Regression1 MSE": rr1_mse_list,
    #                                     "Ridge Regression1 QL": rr1_ql_list,
    #                                     "Optimal log_lambda": rr1_optimal_log_lambda_list})
    # rr1_test_restuls_df = rr1_test_restuls_df.set_index(np.transpose(filenames_nocsv), drop=True)
    # rr1_test_restuls_df.to_csv('Ridge_Regression1_test_MSE_QL.csv')
