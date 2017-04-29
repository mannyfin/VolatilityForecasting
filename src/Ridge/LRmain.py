
"""
Linear regression. calls linear_regression.py and produces plots and outputs.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import linear_regression as lr


def LR(train_set, test_set, warmup_period, name,n_seq, dictlist):
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
        MSE = lr.lin_reg(train_set, n, warmup_period, name=name)[0]
        lr_mse_train_list.append(MSE)

        print( "LR MSE for n=" +str(n ) + " is: " +str(MSE))
    lrdf = pd.DataFrame([n_seq ,lr_mse_train_list] ,index=["n" ,"lr_mse_train"])
    lrdf.to_csv(str(name ) +" lr_mse_train.csv")
    n = lr_mse_train_list.index(min(lr_mse_train_list)) + 1  # add one because index starts at zero

    # lr_optimal_n_list.append(n)
    dictlist['LR']['lr_optimal_n_list'].append(n)


    print( "The smallest n for LR is n=" +str(n))
    figLR = plt.figure(figsize=(8, 6))
    ax_LR = figLR.add_subplot(111)
    ax_LR.plot(range(1, 16), lr_mse_train_list)
    ax_LR.set(title=name.replace(".csv" ,"" ) + ' MSE vs n (warmup: ' +str(warmup_period ) + ')\noptimal n=' +str(n), xlabel='number of regressors', ylabel='MSE')
    plt.savefig(name.replace(".csv" ,"" ) + ' LinearReg MSE vs n_warmup_' +str(warmup_period ) +'.png')

    print('\nTesting ...\n')
    # LR test set. Use the entire training set as the fit for the test set. See code in LR.
    MSE_LR_test, QL_LR_test, ln_SE_LR_test, PredVol_LR_test, b_LR_test, c_LR_test = lr.lin_reg(train_set, n, warmup_period=warmup_period
                                                                                               ,name=name, test=(True, test_set))
    print( "LR(" +str(n ) + ") test MSE: " +str(MSE_LR_test ) + "; test QL: " +str(QL_LR_test))

    # lr_mse_list.append(MSE_LR_test)
    # lr_ql_list.append(QL_LR_test)
    # lr_lnSE_list.append(ln_SE_LR_test)
    # lr_PredVol_list.append(PredVol_LR_test)
    dictlist['LR']['lr_mse_list'].append(MSE_LR_test)
    dictlist['LR']['lr_ql_list'].append(QL_LR_test)
    dictlist['LR']['lr_lnSE_list'].append(ln_SE_LR_test)
    dictlist['LR']['lr_PredVol_list'].append(PredVol_LR_test)

    lr_lnse = dictlist['LR']['lr_lnSE_list'][0]
    lr_predvol = dictlist['LR']['lr_PredVol_list'][0]

    # lr_lnSE_list_df = pd.DataFrame(np.array([lr_lnSE_list[0]]), index=["lr_lnSE"]).transpose()
    # lr_PredVol_list_df = pd.DataFrame(np.array([lr_PredVol_list[0]]), index=["lr_PredVol"]).transpose()

    lr_lnSE_list_df = pd.DataFrame(np.array([lr_lnse]), index=["lr_lnSE"]).transpose()
    lr_PredVol_list_df = pd.DataFrame(np.array([lr_predvol]), index=["lr_PredVol"]).transpose()



    lr_lnSE_list_df.to_csv(str(name ) +"lr_lnSE.csv")
    lr_PredVol_list_df.to_csv(str(name ) +"lr_PredVol.csv")
    #
    # lr_test_restuls_df = pd.DataFrame({"LinearReg MSE":lr_mse_list,
    #                                    "LinearReg QL": lr_ql_list,
    #                                    "Optimal n": lr_optimal_n_list})
    # lr_test_restuls_df = lr_test_restuls_df.set_index(np.transpose(filenames_nocsv), drop=True)
    # lr_test_restuls_df.to_csv('LinearReg_test_MSE_QL_warmup_'+str(warmup_period)+'.csv')


    # return lr_mse_list, lr_ql_list, lr_optimal_n_list
    return dictlist
    # lr_test_restuls_df = pd.DataFrame({"LinearReg MSE":lr_mse_list,
    #                                    "LinearReg QL": lr_ql_list,
    #                                    "Optimal n": lr_optimal_n_list})
    # lr_test_restuls_df = lr_test_restuls_df.set_index(np.transpose(filenames_nocsv), drop=True)
    # lr_test_restuls_df.to_csv('LinearReg_test_MSE_QL_warmup_'+str(warmup_period)+'.csv')