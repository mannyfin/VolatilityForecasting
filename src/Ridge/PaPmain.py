import pandas as pd
import numpy as np
from PastAsPresent import PastAsPresent as pap


def PaP(test_set, name, filenames_nocsv):
    pap_mse_list = []
    pap_ql_list = []
    pap_lnSE_list = []
    pap_PredVol_list = []
    """
            PastAsPresent
    """
    print(str('-') * 24 + "\n\nPerforming PastAsPresent\n\n")

    # PastAsPresent -- Test sample only
    papMSE_test, papQL_test, pap_ln_SE_test ,pap_PredVol_test = pap.tn_pred_tn_plus_1(test_set)
    print("Past as Present MSE: " + str(papMSE_test) + "; QL: " + str(papQL_test))

    pap_mse_list.append(papMSE_test)
    pap_ql_list.append(papQL_test)
    pap_lnSE_list.append(pap_ln_SE_test)
    pap_PredVol_list.append(pap_PredVol_test)

    pap_lnSE_list_df = pd.DataFrame(np.array([pap_lnSE_list[0]]), index=["pap_lnSE"]).transpose()
    pap_PredVol_list_df = pd.DataFrame(np.array([pap_PredVol_list[0]]), index=["pap_PredVol"]).transpose()
    pap_lnSE_list_df.to_csv(str(name ) +" pap_lnSE.csv")
    pap_PredVol_list_df.to_csv(str(name ) +" pap_PredVol.csv")

    return pap_mse_list, pap_ql_list

# pap_test_restuls_df = pd.DataFrame({"PastAsPresent MSE":pap_mse_list,
#                                      "PastAsPresent QL": pap_ql_list})
# pap_test_restuls_df = pap_test_restuls_df.set_index(np.transpose(filenames_nocsv), drop=True)
# pap_test_restuls_df.to_csv('PastAsPresent_test_MSE_QL.csv')