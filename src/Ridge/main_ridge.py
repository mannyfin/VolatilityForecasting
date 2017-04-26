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


dailyvol_zeroes= pd.DataFrame()
filenames = ['AUDUSD.csv']
# filenames = ['AUDUSD.csv', 'CADUSD.csv',  'CHFUSD.csv', 'EURUSD.csv', 'GBPUSD.csv', 'NOKUSD.csv', 'NZDUSD.csv']

# vars
warmup_period = 250

for count, name in enumerate(filenames):
    # initialize some lists
    lr_mse_list = []
    rr_mse_list = []
    brr_mse_list = []
    krr_mse_list = []

    print("Running file: " + str(name))
    #  reads in the files and puts them into dataframes, returns a dataframe called df
    df, df_single_day = rd.read_in_files(name, day=1)
    name = name.split('.')[0]
    # name.append(name)


    # daily_vol_result is the entire vol dataset
    daily_vol_result, daily_ret, daily_vol_zeroes, daily_ret_zeroes = vol.time_vol_calc(df_single_day)

    #  Split the dataset into train and test set
    #  909 is break point for train/test
    train_set, test_set = sd.split_data(dataframe=daily_vol_result, idx=909, reset_index=False)

    """
            PastAsPresent
    """
    print(str('-') * 24 + "\n\nPerforming PastAsPresent\n\n")

    # PastAsPresent -- Test sample only
    papMSE_test, papQL_test, pap_ln_SE_test = pap.tn_pred_tn_plus_1(test_set)
    print("Past as Present MSE: "+str(papMSE_test))

    """
            Linear Regression
    """

    print(str('-') * 28 + "\n\nPerforming Linear Regression\n\n")
    # LR model for 10 regressors on the training set
    print("Training ... \n")
    for n in range(1,11):
        MSE, QL, ln_SE, b, c = lr.lin_reg(train_set, n, warmup_period)
        lr_mse_list.append(MSE)

        print("LR MSE for n="+str(n)+" is: "+str(MSE))

    n = lr_mse_list.index(min(lr_mse_list)) + 1  # add one because index starts at zero
    print("The smallest n for LR is n="+str(n))
    figLR = plt.figure(figsize=(8, 6))
    ax_LR = figLR.add_subplot(111)
    ax_LR.plot(range(1, 11), lr_mse_list)
    ax_LR.set(title='MSE vs n', xlabel='number of regressors', ylabel='MSE')

    print('\nTesting ...\n')
    # LR test set. Use the entire training set as the fit for the test set. See code in LR.
    MSE_LR_test, QL_LR_test, ln_SE_LR_test, b_LR_test, c_LR_test = lr.lin_reg(daily_vol_result, n, warmup_period=warmup_period, test=(True, test_set))

    print("LR(10) MSE in the test set is: "+str(MSE_LR_test))

    """
            Ridge Regression
    """

    print(str('-') * 27 + "\n\nPerforming Ridge Regression\n\n")
    print("Training ... \n")

    # Current status: Working code for train set
    # TODO vary lamda while holding n = 9
    # TODO vary both n and lamda
    for n in range(1,11):
        MSE, QL, ln_SE, b, c = rr.ridge_reg(train_set, n, warmup_period,lamda=1)
        rr_mse_list.append(MSE)

        print("RR MSE for n="+str(n)+" is: "+str(MSE))

    n = rr_mse_list.index(min(rr_mse_list)) + 1  # add one because index starts at zero
    print("The smallest n for RR is n="+str(n))
    print('\nTesting ...\n')

    """
           Bayesian Ridge Regression
    """
    print(str('-') * 36 + "\n\nPerforming Bayesian Ridge Regression\n\n")
    print("Training ... \n")
    # Current status: Working code for train set
    # TODO vary alphas and lamdas while holding n = 9
    # TODO vary both n and lamdas and alphas
    for n in range(1,11):
        MSE, QL, ln_SE, b, c = brr.bayes_ridge_reg(train_set, n, warmup_period, alpha_1=1e-06, alpha_2=1e-06,
                                                   lambda_1=1e-06, lambda_2=1e-06, test=False)
        brr_mse_list.append(MSE)

        print("BRR MSE for n="+str(n)+" is: "+str(MSE))

    n = brr_mse_list.index(min(brr_mse_list)) + 1  # add one because index starts at zero
    print("The smallest n for BRR is n="+str(n))
    print('\nTesting ...\n')

    """
           Kernel Ridge Regression
    """
    print(str('-') * 34 + "\n\nPerforming Kernel Ridge Regression\n\n")
    print("Training ... \n")
    # Current status: Working code for train set
    # TODO vary alphas and lamdas while holding n = 9
    # TODO vary a whole bunch of stuff
    kernels=['linear', 'polynomial', 'sigmoid', 'rbf', 'laplacian' ]  #  chi2

    for kernel in kernels:
        for n in range(1,11):
            MSE, QL, ln_SE = krr.kernel_ridge_reg(train_set, n, warmup_period, alpha=1,coef0=1, degree=3, kernel=kernel, test=False)
            krr_mse_list.append(MSE)
            print("KRR MSE for n=" + str(n) + " and kernel=" + kernel + " is: " + str(MSE))

        n = krr_mse_list.index(min(krr_mse_list)) + 1  # add one because index starts at zero
        print("The smallest n for KRR is n=" + str(n) + " for kernel = " + kernel)
        print("\nTraining ... \n")

    print('\nTesting ...\n')

    # feel free to put a breakpoint in the line below...
    print('hi')

    """
    Notes:
    Test set code is not written yet for RR, BRR, or KRR...
    
    kernel = polynomial and sigmoid run very slow.
    kernel = rbf, linear, and laplacian run pretty fast.
    kernel = sigmoid MSE is terrible on the train set
    
    kernel = chi2 gives me an error...see below.
    
OUTPUT

Running file: AUDUSD.csv
------------------------

Performing PastAsPresent


Past as Present MSE: 0.000677190421995
----------------------------

Performing Linear Regression


Training ... 

LR MSE for n=1 is: 0.00101481488567
LR MSE for n=2 is: 0.000936783794972
LR MSE for n=3 is: 0.00094051482296
LR MSE for n=4 is: 0.000933448615908
LR MSE for n=5 is: 0.000924728539503
LR MSE for n=6 is: 0.000927502269249
LR MSE for n=7 is: 0.000922801269685
LR MSE for n=8 is: 0.000923686734076
LR MSE for n=9 is: 0.000922411265635
LR MSE for n=10 is: 0.000920593392994
The smallest n for LR is n=10

Testing ...

LR(10) MSE in the test set is: 0.000479657855634
---------------------------

Performing Ridge Regression


Training ... 

RR MSE for n=1 is: 0.00101108302058
RR MSE for n=2 is: 0.000935202214518
RR MSE for n=3 is: 0.00093965784718
RR MSE for n=4 is: 0.000933161381345
RR MSE for n=5 is: 0.000924425052345
RR MSE for n=6 is: 0.000926757751116
RR MSE for n=7 is: 0.000921685220548
RR MSE for n=8 is: 0.000921021108621
RR MSE for n=9 is: 0.000919130846668
RR MSE for n=10 is: 0.00091687591175
The smallest n for RR is n=10

Testing ...

------------------------------------

Performing Bayesian Ridge Regression


Training ... 

BRR MSE for n=1 is: 0.00101460821754
BRR MSE for n=2 is: 0.00093652969911
BRR MSE for n=3 is: 0.000940334231144
BRR MSE for n=4 is: 0.000933512500354
BRR MSE for n=5 is: 0.000924913240712
BRR MSE for n=6 is: 0.000927571723697
BRR MSE for n=7 is: 0.000922660397337
BRR MSE for n=8 is: 0.000922534610268
BRR MSE for n=9 is: 0.000920564798842
BRR MSE for n=10 is: 0.000918167236629
The smallest n for BRR is n=10

Testing ...

------------------------------------

Performing Kernel Ridge Regression


Training ... 

KRR MSE for n=1 and kernel=linear is: 0.00111179180933
KRR MSE for n=2 and kernel=linear is: 0.000979549863475
KRR MSE for n=3 and kernel=linear is: 0.000975536412152
KRR MSE for n=4 and kernel=linear is: 0.00095877615691
KRR MSE for n=5 and kernel=linear is: 0.000943299920845
KRR MSE for n=6 and kernel=linear is: 0.000943798936509
KRR MSE for n=7 and kernel=linear is: 0.000935082933737
KRR MSE for n=8 and kernel=linear is: 0.000931869221408
KRR MSE for n=9 and kernel=linear is: 0.000928328518061
KRR MSE for n=10 and kernel=linear is: 0.000924466534838
The smallest n for KRR is n=10 for kernel = linear
KRR MSE for n=1 and kernel=polynomial is: 0.001005586998
KRR MSE for n=2 and kernel=polynomial is: 0.00094359561238
KRR MSE for n=3 and kernel=polynomial is: 0.000947598939324
KRR MSE for n=4 and kernel=polynomial is: 0.000944340887012
KRR MSE for n=5 and kernel=polynomial is: 0.000938939435993
KRR MSE for n=6 and kernel=polynomial is: 0.000944393434271
KRR MSE for n=7 and kernel=polynomial is: 0.000942598229622
KRR MSE for n=8 and kernel=polynomial is: 0.000932349703323
KRR MSE for n=9 and kernel=polynomial is: 0.000938533096635
KRR MSE for n=10 and kernel=polynomial is: 0.000938961255562
The smallest n for KRR is n=10 for kernel = polynomial
I:\Anaconda3\lib\site-packages\sklearn\linear_model\ridge.py:154: UserWarning: Singular matrix in solving dual problem. Using least-squares solution instead.
  warnings.warn("Singular matrix in solving dual problem. Using "
KRR MSE for n=1 and kernel=sigmoid is: 0.00193776225919
KRR MSE for n=2 and kernel=sigmoid is: 0.0016905076853
KRR MSE for n=3 and kernel=sigmoid is: 0.0016968621607
KRR MSE for n=4 and kernel=sigmoid is: 0.00165115695034
KRR MSE for n=5 and kernel=sigmoid is: 0.00165187206087
KRR MSE for n=6 and kernel=sigmoid is: 0.00165901101048
KRR MSE for n=7 and kernel=sigmoid is: 0.00164985139608
KRR MSE for n=8 and kernel=sigmoid is: 0.00165951812041
KRR MSE for n=9 and kernel=sigmoid is: 0.00165494735817
KRR MSE for n=10 and kernel=sigmoid is: 0.00165877382423
The smallest n for KRR is n=10 for kernel = sigmoid
KRR MSE for n=1 and kernel=rbf is: 0.0010157983252
KRR MSE for n=2 and kernel=rbf is: 0.000985750399317
KRR MSE for n=3 and kernel=rbf is: 0.000988171847439
KRR MSE for n=4 and kernel=rbf is: 0.000983524455797
KRR MSE for n=5 and kernel=rbf is: 0.00097859412136
KRR MSE for n=6 and kernel=rbf is: 0.000995950390833
KRR MSE for n=7 and kernel=rbf is: 0.00099605810568
KRR MSE for n=8 and kernel=rbf is: 0.000984774257374
KRR MSE for n=9 and kernel=rbf is: 0.000980194358956
KRR MSE for n=10 and kernel=rbf is: 0.000972685010345
The smallest n for KRR is n=10 for kernel = rbf
KRR MSE for n=1 and kernel=laplacian is: 0.00105966151203
KRR MSE for n=2 and kernel=laplacian is: 0.000986192980385
KRR MSE for n=3 and kernel=laplacian is: 0.000994942240928
KRR MSE for n=4 and kernel=laplacian is: 0.000981252894236
KRR MSE for n=5 and kernel=laplacian is: 0.000979944175186
KRR MSE for n=6 and kernel=laplacian is: 0.000987171570476
KRR MSE for n=7 and kernel=laplacian is: 0.000983416699001
KRR MSE for n=8 and kernel=laplacian is: 0.00097829487795
KRR MSE for n=9 and kernel=laplacian is: 0.000972614451636
Traceback (most recent call last):
KRR MSE for n=10 and kernel=laplacian is: 0.000969209909576
  File "I:\PyCharm\PyCharm 2016.3.2\helpers\pydev\pydevd.py", line 1578, in <module>
The smallest n for KRR is n=10 for kernel = laplacian
    globals = debugger.run(setup['file'], None, None, is_module)
  File "I:\PyCharm\PyCharm 2016.3.2\helpers\pydev\pydevd.py", line 1015, in run
    pydev_imports.execfile(file, globals, locals)  # execute the script
  File "I:\PyCharm\PyCharm 2016.3.2\helpers\pydev\_pydev_imps\_pydev_execfile.py", line 18, in execfile
    exec(compile(contents+"\n", file, 'exec'), glob, loc)
  File "C:/Users/Manolis1/Desktop/FIN580/Homework1/VolatilityForecasting/src/Ridge/main_ridge.py", line 127, in <module>
    MSE, QL, ln_SE = krr.kernel_ridge_reg(train_set, n, warmup_period, alpha=1,coef0=1, degree=3, kernel=kernel, test=False)
  File "C:/Users/Manolis1/Desktop/FIN580/Homework1/VolatilityForecasting/src/Ridge\KernelRidgeRegression.py", line 31, in kernel_ridge_reg
    A.fit(xstacked, y)
  File "I:\Anaconda3\lib\site-packages\sklearn\kernel_ridge.py", line 149, in fit
    K = self._get_kernel(X)
  File "I:\Anaconda3\lib\site-packages\sklearn\kernel_ridge.py", line 121, in _get_kernel
    filter_params=True, **params)
  File "I:\Anaconda3\lib\site-packages\sklearn\metrics\pairwise.py", line 1399, in pairwise_kernels
    return _parallel_pairwise(X, Y, func, n_jobs, **kwds)
  File "I:\Anaconda3\lib\site-packages\sklearn\metrics\pairwise.py", line 1083, in _parallel_pairwise
    return func(X, Y, **kwds)
  File "I:\Anaconda3\lib\site-packages\sklearn\metrics\pairwise.py", line 1027, in chi2_kernel
    K = additive_chi2_kernel(X, Y)
  File "I:\Anaconda3\lib\site-packages\sklearn\metrics\pairwise.py", line 975, in additive_chi2_kernel
    raise ValueError("X contains negative values.")
ValueError: X contains negative values.

Process finished with exit code 1

    """