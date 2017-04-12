
#  if you dont have this installed then here's how to install the module:
#  conda install -c https://conda.binstar.org/bashtage arch

from read_in_files import read_in_files
from NumDaysWeeksMonths import NumDaysWeeksMonths
from Volatility import *
from linear_regression import *
import matplotlib.pyplot as plt
from PastAsPresent import *
# from VAR import *
from tablegen import tablegen
from garch_pq_model import GarchModel as gm
# from arch_q_model import ArchModelQ as am
import numpy as np
import pandas as pd
from KNN3_1 import KNN
from function_runs import *
import os
from VAR2 import *
from multiprocessing import Pool
import multiprocessing
import time
import matplotlib.backends.backend_pdf
import prediction_output


name = ['AUDUSD.csv', 'CADUSD.csv',  'CHFUSD.csv', 'EURUSD.csv', 'GBPUSD.csv', 'NOKUSD.csv', 'NZDUSD.csv']
# name = ['CADUSD.csv'] #,  'CHFUSD.csv', 'EURUSD.csv', 'GBPUSD.csv', 'NOKUSD.csv', 'NZDUSD.csv']

KNN_test = []

# for filenames in filenamesx:
dailyvol_zeroes= pd.DataFrame()
weeklyvol_zeroes= pd.DataFrame()
monthlyvol_zeroes= pd.DataFrame()
dailyret_zeroes= pd.DataFrame()
weeklyret_zeroes= pd.DataFrame()
monthlyret_zeroes = pd.DataFrame()
Daily_list = list()
namelist = list()
# for count, names in enumerate(filenamesx):

file_results_list =[]
test_set_results_list=[]

def multiknn(name):
    KNN_test = []
    MSE_training_all = []
    QL_training_all = []
    # for filenames in filenamesx:
    dailyvol_zeroes = pd.DataFrame()
    weeklyvol_zeroes = pd.DataFrame()
    monthlyvol_zeroes = pd.DataFrame()
    dailyret_zeroes = pd.DataFrame()
    weeklyret_zeroes = pd.DataFrame()
    monthlyret_zeroes = pd.DataFrame()
    Daily_list = list()
    namelist = list()
    all_results_plot =[]
    #  reads in the files and puts them into dataframes, returns a dataframe called df
    df, df_single_day, df_single_week, df_single_month = read_in_files(names)
    # THE LINE BELOW IS NOT REALLY NEEDED IF SCALING IS CONSTANT
    # days_weeks_months, num_days_per_year, num_weeks_per_year, num_months_per_year = NumDaysWeeksMonths(df=df)
    # We use this line below for the names of the graph
    name = name.split('.')[0]
    namelist.append(names)
    print("Running file: " + str(name))
    warmup_period = 100
    daily_vol_result, daily_ret, daily_vol_zeroes, daily_ret_zeroes = time_vol_calc(df_single_day)
    # weekly_vol_result, weekly_ret, weekly_vol_zeroes, weekly_ret_zeroes = time_vol_calc(df_single_week)
    # monthly_vol_result, monthly_ret, monthly_vol_zeroes, monthly_ret_zeroes = time_vol_calc(df_single_month)

    # be careful of the underscore, _
    dailyvol_zeroes = pd.concat([dailyvol_zeroes, daily_vol_zeroes], axis=1)

    # dailyvol_zeroes = pd.concat([dailyvol_zeroes, daily_vol_zeroes['Volatility_Time']], axis=1)

    dailyvol_zeroes.rename(columns={'Volatility_Time': name}, inplace=True)

    # be careful of the underscore, _
    dailyret_zeroes = pd.concat([dailyret_zeroes, daily_ret_zeroes], axis=1)

    dailyret_zeroes.rename(columns={'Return_Time': name}, inplace=True)

    # does not have zeroes
    daily_vol_combined = dailyvol_zeroes[(dailyvol_zeroes != 0).all(1)]

    try:
        dates = daily_vol_combined.Date.loc[:, ~daily_vol_combined.Date.columns.duplicated()].reset_index()
        daily_vol_combined.drop('Date', axis=1, inplace=True)
        daily_vol_combined=daily_vol_combined.apply(pd.to_numeric)

    except:
        dates = daily_vol_combined.Date
    # drop duplicate columns

    # weekly_vol_combined = weeklyvol_zeroes[(weeklyvol_zeroes != 0).all(1)]
    # monthly_vol_combined = monthlyvol_zeroes[(monthlyvol_zeroes != 0).all(1)]
    """I am referencing the $Time$vol_zeroes variable in the lines below because there are (or could be) days where the ret
      is zero"""
    daily_ret_combined = dailyret_zeroes[(dailyvol_zeroes != 0).all(1)]
    # weekly_ret_combined = weeklyret_zeroes[(weeklyvol_zeroes != 0).all(1)]
    # monthly_ret_combined = monthlyret_zeroes[(monthlyvol_zeroes != 0).all(1)]

    # We need to reset index
    daily_vol_combined.reset_index(drop=True, inplace=True)
    # weekly_vol_combined.reset_index(drop=True, inplace=True)
    # monthly_vol_combined.reset_index(drop=True, inplace=True)
    daily_ret_combined.reset_index(drop=True, inplace=True)

    p=3
    # use this below
    fc = FunctionCalls()
    # xmat = pd.DataFrame([sum([daily_vol_combined[currency].loc[i+p-1:i:-1].as_matrix().tolist() for currency in daily_vol_combined.keys()],[]) for i in range(len(daily_vol_combined)-p)])


    # 11/1
    training_sample, test_sample = daily_vol_combined.iloc[0:910].copy(), daily_vol_combined.iloc[910:].copy()

    # remove the Date column from the data before passing. Separate it out out
    training_sample.reset_index(drop=True)
    training_date = pd.DataFrame(training_sample["Date"])
    training_sample.drop('Date', axis=1, inplace=True)

    test_sample.reset_index(drop=True)
    test_date = pd.DataFrame(test_sample['Date'])
    test_sample.drop('Date', axis=1, inplace=True)

    if not os.path.exists(name):
        os.mkdir(name)
    os.chdir(name)
    training = True

    options = {'zero': 'method0',
               'one': 'method1'
               }
    # method_choice is used for pickout out specific KNN tests one may wish to run
    # to run multiple methods, do something like this
    method_choice = [options['zero'], options['one']]


    # method_choice = [options['one']]
    k=20



    if options['zero'] in method_choice:
        # if training:
        if not os.path.exists(name + '_KNN_Train0'):
            os.mkdir(name + '_KNN_Train0')
        os.chdir(name + '_KNN_Train0')
        plt.close()
        # method 0
        KNN_training = [[fc.function_runs(dates=training_date, filename=str(name)+' Single Knn',
                         stringinput='Daily', warmup=warmup, input_data=training_sample, k_nn=[i], options='0-train')
                         for i in np.arange(1, k+1)] for warmup in [100]]  #  200,500)]

        MSE_training = pd.Series([KNN_training[0][i].MSE[name] for i in range(0, len(KNN_training[0]))],
                                 index=np.linspace(1, k, k, dtype=int))
        #  Organize MSE results from KNN_training
        MSE_training_all.append(MSE_training)

        #  Organize QL results from KNN_training
        QL_training = pd.Series([KNN_training[0][i].QL[name] for i in range(0, len(KNN_training[0]))],
                                index=np.linspace(1, k, k, dtype=int))

        # QL_training = pd.Series([KNN_training[0][i].QL.loc['QL'] for i in range(0,len(KNN_training[0]))], index=np.linspace(1,20,20, dtype=int))
        QL_training_all.append(QL_training)


        # combine results
        train_k = pd.DataFrame([MSE_training, QL_training], index=['MSE', 'QL']).transpose()
        train_k.index.names = ['k']

        # save output to excel file
        writer = pd.ExcelWriter(str(name)+' KNN_train_output.xlsx')
        train_k.to_excel(writer, 'Train results')
        writer.save()

        # gives the k-value of the lowest MSE
        k_star = MSE_training.idxmin()
        plttitle = name+' MSE vs k'
        ax = MSE_training.plot()
        ax.set(xlabel='k', ylabel='MSE', title=plttitle, xticks=range(0, 20+1,2))
        plt.plot(k_star, MSE_training.min(), '*', markersize=10)
        plt.savefig(plttitle+'.png')
        plt.close()
        # append to output
        file_results_list.append([name, k_star, MSE_training.min(), QL_training.idxmin(), QL_training.min()])

        os.chdir('..')
        if not os.path.exists(name + '_KNN_Test0'):
            os.mkdir(name + '_KNN_Test0')
        os.chdir(name + '_KNN_Test0')
        # test sample
        KNN_test.append(fc.function_runs(dates=test_date, filename=name, stringinput='Daily', warmup=training_sample,
                                         input_data=test_sample, k_nn=[12], options='0-test'))

        all_results_plot.append(pd.read_csv('test_prediction_0-test.csv', index_col='Unnamed: 0', names=['Unnamed: 0',options['zero']]))
        test_set_results_list.append([name, 12, KNN_test[0].loc[name][0],KNN_test[0].loc[name][1]])
        os.chdir('..')

    if options['one'] in method_choice:
        if not os.path.exists(name + '_KNN_Train1'):
            os.mkdir(name + '_KNN_Train1')
        os.chdir(name + '_KNN_Train1')
        plt.close()

        # method 0
        # KNN_training = [[fc.function_runs(dates=training_date, filename=str(name)+' Single Knn',
        #                  stringinput='Daily', warmup=warmup, input_data=training_sample.astype('float64'), k_nn=[i], options='0-train')
        #                  for i in np.arange(1, k+1)] for warmup in [100]]

        #method1
        KNN_training_1 = [[fc.function_runs(dates=training_date, filename=str(name)+' Single Knn',
                         stringinput='Daily', warmup=warmup, input_data=training_sample.astype('float64'), k_nn=[i], options='1-train')
                         for i in np.arange(1, k+1)] for warmup in [100]]  #  200,500)]

        #  Organize MSE results from KNN_training
        MSE_training = pd.Series([KNN_training_1[0][i].MSE[name] for i in range(0, len(KNN_training_1[0]))],
                                 index=np.linspace(1, k, k, dtype=int))

        MSE_training_all.append(MSE_training)
        #  Organize QL results from KNN_training
        # QL_training = pd.Series([KNN_training[0][i].QL.loc['QL'] for i in range(0,len(KNN_training[0]))], index=np.linspace(1,20,20, dtype=int))
        QL_training =  pd.Series([KNN_training_1[0][i].QL[name] for i in range(0, len(KNN_training_1[0]))],
                                index=np.linspace(1, k, k, dtype=int))
        QL_training.append(QL_training)
        # combine results
        train_k = pd.DataFrame([MSE_training, QL_training], index=['MSE', 'QL']).transpose()
        train_k.index.names = ['k']

        # save output to excel file
        writer = pd.ExcelWriter(str(name)+' KNN_train_output.xlsx')
        train_k.to_excel(writer, 'Train results')
        writer.save()

        # gives the k-value of the lowest MSE
        k_star = MSE_training.idxmin()
        plttitle = name+' MSE vs k'
        ax = MSE_training.plot()
        ax.set(xlabel='k', ylabel='MSE', title=plttitle, xticks=range(0, 20+1,2))
        ax.plot(k_star.astype(int), MSE_training.min().astype('float64'), '*', markersize=10)
        plt.savefig(plttitle+'.png')
        plt.close()
        # append to output
        file_results_list.append([name, k_star, MSE_training.min(), QL_training.idxmin(), QL_training.min()])

        os.chdir('..')
        if not os.path.exists(name + '_KNN_Test1'):
            os.mkdir(name + '_KNN_Test1')
        # test sample
        KNN_test.append(fc.function_runs(dates=test_date, filename=name, stringinput='Daily', warmup=training_sample,
                                         input_data=test_sample, k_nn=[12], options='1-test'))

        all_results_plot.append(pd.read_csv('test_prediction_1-test.csv', index_col='Unnamed: 0', names=['Unnamed: 0',options['one']]))
        test_set_results_list.append([name, 12, KNN_test[0].loc[name][0], KNN_test[0].loc[name][1]])
        os.chdir('..')

    # else:
    #         KNN_test.append(fc.function_runs(dates=test_date, filename=name, stringinput='Daily', warmup=100, input_data=test_sample, k_nn=[20]))
    # plt.show()
    plt.close()
    allmethods = pd.DataFrame(MSE_training_all).transpose()
    allmethods.columns = method_choice
    writer = pd.ExcelWriter('MSE_Train_results.xlsx')
    allmethods.to_excel(writer, 'Combined MSE Train results')
    writer.save()

    # plot results
    all_results_plot.append(test_sample[name].reset_index().drop('index', axis=1))
    all_results_plot.append(test_date['Date'].reset_index().drop('index', axis=1))
    df_combined_pred = pd.concat(all_results_plot, axis=1)
    df_combined_pred = df_combined_pred.dropna(axis=0).reset_index().drop('index', axis=1)
    df_combined_pred = df_combined_pred.set_index('Date')
    ax=df_combined_pred.plot(kind='line', figsize=(12, 7))
    ax.set(ylabel='log(RV)', title=name + ' prediction methods vs observed')
    plt.savefig(name + ' prediction methods vs observed'+'.png')
    plt.close()

    lnSE = df_combined_pred.sub(df_combined_pred[name].astype('float64'), axis=0).astype('float64').apply(np.square).apply(np.log)
    lnSE = lnSE.drop(name, axis=1)
    ax1 = lnSE.plot(kind='line', figsize=(12, 7))
    ax1.set(ylabel='ln(SE)', title=name + ' ln(SE)')
    plt.savefig(name + ' ln(SE)'+'.png')
    plt.close()
    #
    # if len(KNN_training) != 1:
    #     ks = np.arange(2,10)
    #
    #     # were indented....
    #     warmups = np.arange(50,200,50)
    #
    #     k,w = np.meshgrid(ks,warmups)
    #     SEkw = np.array([ [ KNN_training[warmup][i-2]['MSE'][0] for i in np.arange(2,10) ] for warmup in np.arange(len(np.arange(50,200,50)))])
    #     k,w = np.meshgrid(ks,warmups)
    #
    #     from mpl_toolkits.mplot3d import Axes3D
    #     from matplotlib import cm
    #     matplotlib.style.use('ggplot')
    #
    #     fig = plt.figure()
    #     ax = fig.gca(projection='3d')
    #     sur=ax.plot_surface(k,w,SEkw,cmap=cm.coolwarm,linewidth=0,antialiased=False,rstride=1, cstride=1)
    #     fig.colorbar(sur,shrink=0.5, aspect=5)
    #
    #     ax.set_xlabel('k')
    #     ax.set_ylabel('warmup')
    #     ax.set_zlabel('SE')
    #     plt.title(str(names[0]))
    #     plt.show()

    os.chdir('..')

# """Output multiple plots into a pdf file"""
# pdf = matplotlib.backends.backend_pdf.PdfPages(names+".pdf")
# for fig in range(1, 3*count+3+1):
#     pdf.savefig( fig, dpi=1200 )
# pdf.close()
#
#
for names in name:
    multiknn(names)
writer=pd.ExcelWriter('KNNoutput.xlsx')
best_knn_training = pd.DataFrame(file_results_list, columns=['File', 'k*_MSE', 'MSE', 'k*_QL', 'QL'])
best_knn_training.to_excel(writer,'Train results')
test_results = pd.DataFrame(test_set_results_list, columns=['File', 'k', 'MSE', 'QL'])
test_results.to_excel(writer,'Test results')

writer.save()

# if __name__ == '__main__':
#
#     # p = Pool(processes = len(names))
#     p = Pool(processes=len(name))
#     start = time.time()
#     async_result = p.map_async(multiknn, name)
#     # for file in names:
#     #     # p.apply_async(multip, [file])
#     #     p.map(multip, [file])
#     #     # p.start()
#     p.close()
#     p.join()
#     print("Complete")
#     end = time.time()
#     print('total time (s)= ' + str(end-start))
#     print("Complete")
