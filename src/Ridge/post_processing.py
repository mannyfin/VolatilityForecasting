import pandas as pd
import numpy as np


def PaP_post_process(dictlist, names):
    """
            Past as Present
    """
    pap_test_results_df = pd.DataFrame({"PaP MSE": dictlist['PaP']['pap_mse_list'],
                                        "PaP QL": dictlist['PaP']['pap_ql_list']})

    pap_test_results_df = pap_test_results_df.set_index(np.transpose(names), drop=True)
    pap_test_results_df.to_csv('PaP_test_MSE_QL.csv')

    # lnSE to csv
    pap_lnSE = pd.DataFrame(dictlist['PaP']['pap_lnSE_list']).T
    pap_lnSE.to_csv('PaP_lnSE_list.csv')

    # Predicted vol for each file to csv
    pap_PredVol = pd.DataFrame(dictlist['PaP']['pap_PredVol_list']).T
    pap_PredVol.to_csv('PaP_predictedVol.csv')


def KRR_post_process(dictlist, names, kernels):

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

        # added during run for combine all files
        krr_test_results = pd.concat(
            [pd.concat(dictlist['KRR'][kernel]['krr1_mse_list']), pd.concat(dictlist['KRR'][kernel]['krr1_ql_list']),
             pd.DataFrame({'optimal_log_alpha': dictlist['KRR'][kernel]['krr_optimal_log_alpha_list']})], join='outer')
        krr_test_results.to_csv(str(kernel)+'_KRR_test_MSE_QL.csv')

        # dont use the below stuff
        krr_test_results_df = krr_test_results_df.set_index(np.transpose(names), drop=True)
        krr_test_results_df.to_csv(str(kernel)+'_KRR_test_MSE_QL.csv')


        # lnSE to csv for all files
        krr1_lnSE = pd.DataFrame(dictlist['KRR'][kernel]['krr1_lnSE_list']).T
        krr1_lnSE.to_csv(str(kernel) + '_KRR1_lnSE_list.csv')

        # Predicted vol for each file to csv
        krr1_PredVol = pd.DataFrame(dictlist['KRR'][kernel]['krr1_PredVol_list']).T
        krr1_PredVol.to_csv(str(kernel) + '_KRR1_predictedVol.csv')

def BRR_post_process(dictlist, names):
    """
            Bayesian Ridge Regression
    """
    brr_test_results_df = pd.DataFrame({"BRR MSE": dictlist['BRR']['brr1_mse_list'],
                                        "BRR QL": dictlist['BRR']['brr1_ql_list'],
                                        "Optimal alpha1": dictlist['BRR']['brr_optimal_log_alpha1_list'],
                                        "Optimal alpha2": dictlist['BRR']['brr_optimal_log_alpha2_list'],
                                        "Optimal lambda1": dictlist['BRR']['brr_optimal_log_lambda1_list'],
                                        "Optimal lambda2": dictlist['BRR']['brr_optimal_log_lambda2_list']})

    brr_test_results_df = brr_test_results_df.set_index(np.transpose(names), drop=True)
    brr_test_results_df.to_csv('BRR_test_MSE_QL.csv')

    # lnSE to csv
    brr1_lnSE = pd.DataFrame(dictlist['BRR']['brr1_lnSE_list']).T
    brr1_lnSE.to_csv('BRR1_lnSE_list.csv')

    # Predicted vol for each file to csv
    brr1_PredVol = pd.DataFrame(dictlist['BRR']['brr1_PredVol_list']).T
    brr1_PredVol.to_csv('BRR1_predictedVol.csv')

