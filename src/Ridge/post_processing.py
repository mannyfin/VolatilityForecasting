import pandas as pd
import numpy as np


def KRR_post_process(dictlist,names,kernels):

    # this is probably the list of kernels
    for kernel in kernels:
        if kernel is 'polynomial':
            krr_test_results_df = pd.DataFrame({"KRR MSE": dictlist['KRR'][kernel]['krr1_mse_list'],
                                                "KRR QL": dictlist['KRR'][kernel]['krr1_ql_list'],
                                                "Optimal alpha": dictlist['KRR'][kernel]['krr_optimal_log_alpha_list'],
                                                "Optimal coef0": dictlist['KRR'][kernel]['krr_optimal_log_coef0_list'],
                                                "Optimal degree": dictlist['KRR'][kernel]['krr_optimal_degree_list']})
            # lnSE to csv for all files
            krr1_lnSE = pd.DataFrame(dictlist['KRR'][kernel]['krr1_lnSE_list']).T
            krr1_lnSE.to_csv(str(kernel)+'_KRR1_lnSE_list.csv')

            # Predicted vol for each file to csv
            krr1_PredVol = pd.DataFrame(dictlist['KRR'][kernel]['krr1_PredVol_list']).T
            krr1_PredVol.to_csv(str(kernel)+'_KRR1_predictedVol.csv')

        else:
            krr_test_results_df = pd.DataFrame({"KRR MSE": dictlist['KRR'][kernel]['krr1_mse_list'],
                                                "KRR QL": dictlist['KRR'][kernel]['krr1_ql_list'],
                                                "Optimal alpha": dictlist['KRR'][kernel]['krr_optimal_log_alpha_list']})
            # lnSE to csv for all files
            krr1_lnSE = pd.DataFrame(dictlist['KRR'][kernel]['krr1_lnSE_list']).T
            krr1_lnSE.to_csv('KRR1_lnSE_list.csv')

            # Predicted vol for each file to csv
            krr1_PredVol = pd.DataFrame(dictlist['KRR'][kernel]['krr1_PredVol_list']).T
            krr1_PredVol.to_csv('KRR1_predictedVol.csv')



        krr_test_results_df = krr_test_results_df.set_index(np.transpose(names), drop=True)
        krr_test_results_df.to_csv(str(kernel)+'_KRR_test_MSE_QL.csv')


        # lnSE to csv for all files
        krr1_lnSE = pd.DataFrame(dictlist['KRR'][kernel]['krr1_lnSE_list']).T
        krr1_lnSE.to_csv(str(kernel) + '_KRR1_lnSE_list.csv')

        # Predicted vol for each file to csv
        krr1_PredVol = pd.DataFrame(dictlist['KRR'][kernel]['krr1_PredVol_list']).T
        krr1_PredVol.to_csv(str(kernel) + '_KRR1_predictedVol.csv')


