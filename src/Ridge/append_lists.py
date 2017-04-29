def append_outputs(mylists):
    appenders = range(len(mylists))
    for x, lst in zip(appenders, mylists):
        lst.append(x)


def initialize_lists(pap=0, lr =0, rr1 = 0, rr2 = 0, brr = 0, krr = 0):
    "goal: find total number of lists and create mylist of len = nuumlists"
    mylist = []
    numlists=0

    if pap is 1:
        pap_mse_list = []
        pap_ql_list = []
        pap_lnSE_list = []
        pap_PredVol_list = []

        numlists += len(pap_mse_list,pap_ql_list,pap_lnSE_list,pap_PredVol_list)
    if lr is 1:
        lr_optimal_n_list = []
        lr_mse_list = []
        lr_ql_list = []
        lr_lnSE_list = []
        lr_PredVol_list = []

        numlists += len(lr_optimal_n_list, lr_mse_list, lr_ql_list, lr_lnSE_list, lr_PredVol_list)

    if rr1 is 1:
        rr1_mse_list = []
        rr1_ql_list = []
        rr1_lnSE_list = []
        rr1_PredVol_list = []
        rr1_optimal_log_lambda_list = []

        numlists += len(rr1_mse_list, rr1_ql_list, rr1_lnSE_list, rr1_PredVol_list,rr1_optimal_log_lambda_list)

    if rr2 is 1:

        rr2_mse_list = []
        rr2_ql_list = []
        rr2_lnSE_list = []
        rr2_PredVol_list = []
        rr2_optimal_n_list = []
        rr2_optimal_log_lambda_list = []

        numlists += len(rr2_mse_list, rr2_ql_list, rr2_lnSE_list,rr2_PredVol_list,rr2_optimal_n_list,rr2_optimal_log_lambda_list)
    if brr is 1:

        brr_mse_list = []
        brr_ql_list = []
        brr_lnSE_list = []
        brr_PredVol_list = []
        brr_optimal_n_list = []
        brr_optimal_log_alpha1_list = []
        brr_optimal_log_alpha2_list = []
        brr_optimal_log_lambda1_list = []
        brr_optimal_log_lambda2_list = []

        numlists +=9 # len(brr_mse_list, brr_ql_list,brr_lnSE_list, brr_optimal_n_list, brr_optimal_log_alpha1_list)
    if krr is 1:
        "does not support individual kernels at this time (or ever)"
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

        numlists +=37
    mylist = [[] for _ in range(numlists)]
    return mylist