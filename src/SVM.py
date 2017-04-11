import numpy as np
from sklearn.linear_model import LogisticRegression as Logit
from Performance_Measure import *
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC as SVC


def SVM_Predict_y_delta_star_training(train_sample, forecaster, model ,C, p=None, q=None):
    """
    :param train_sample: could be train_sample_daily or train_sample_weekly (obtained in main)
    :param forecaster: forecaster could take values 1, 2, 3, 4, 5 or 6
    :param model: "SVM","KernelSVM_rbf","KernelSVM_sigmoid"
    :param C: margin in the SVM model, needed to be optimized
    :param p: p is a parameter in forecaster 3 and forecaster 4 
    :param q: q is a parameter in forecaster 4    
    :return: the predicted y label, benchmark_delta_star (the benchmark delta_star) and the fitted model in the training sample
    """
    if model == "SVM":
        Model = SVC(C, kernel='linear')
    elif model == "KernelSVM_rbf":
        Model = SVC(C, kernel='rbf')
    elif model == "KernelSVM_sigmoid":
        Model = SVC(C, kernel='sigmoid')

    if forecaster == 1:
        train_sample_new = train_sample.copy()
        fittedModel = Model.fit(np.array(train_sample_new.vol_now).reshape(len(train_sample_new.vol_now),1),
                                np.array(train_sample_new.label))
        predicted_y_train = fittedModel.predict(np.array(train_sample_new.vol_now).reshape(len(train_sample_new.vol_now),1))
    elif forecaster == 2:
        train_sample_new = train_sample.copy()
        inputdf = train_sample_new[['vol_now', 'ret_now']]
        fittedModel = Model.fit(np.array(inputdf).reshape(len(inputdf),2),
                                np.array(train_sample_new.label))
        predicted_y_train = fittedModel.predict(np.array(inputdf).reshape(len(inputdf),2))
    elif forecaster == 3:
        train_sample_new = train_sample.copy()
        train_sample_new["E_p"] = train_sample_new["vol_now"].rolling(p).mean()
        train_sample_new.dropna(inplace=True, axis=0)

        fittedModel = Model.fit(np.array(train_sample_new.E_p).reshape(len(train_sample_new.E_p), 1),
                                np.array(train_sample_new.label))
        predicted_y_train = fittedModel.predict(np.array(train_sample_new.E_p).reshape(len(train_sample_new.E_p), 1))
    elif forecaster == 4:
        train_sample_new = train_sample.copy()
        train_sample_new["E_p"] = train_sample_new["vol_now"].rolling(p).mean()
        train_sample_new["E_q"] = train_sample_new["ret_now"].rolling(q).mean()
        train_sample_new.dropna(inplace=True, axis=0)

        inputdf = train_sample_new[['E_p', 'E_q']]
        fittedModel = Model.fit(np.array(inputdf).reshape(len(inputdf), 2),
                                np.array(train_sample_new.label))
        predicted_y_train = fittedModel.predict(np.array(inputdf).reshape(len(inputdf), 2))
    elif forecaster == 5:
        train_sample_new = train_sample.copy()
        inputdf = train_sample_new[['vol_now','ret_now', 'vol_sqr_now']]
        fittedModel = Model.fit(np.array(inputdf).reshape(len(inputdf),3),
                                np.array(train_sample_new.label))
        predicted_y_train = fittedModel.predict(np.array(inputdf).reshape(len(inputdf),3))
    elif forecaster == 6:
        train_sample_new = train_sample.copy()
        inputdf = train_sample_new[['vol_now',  'ret_now','volxret_now','vol_sqr_now']]
        fittedModel = Model.fit(np.array(inputdf).reshape(len(inputdf),4),
                                np.array(train_sample_new.label))
        predicted_y_train = fittedModel.predict(np.array(inputdf).reshape(len(inputdf),4))

    vol_today = np.array(train_sample_new.vol_now).reshape(len(train_sample_new.vol_now), 1)
    vol_tmr = np.array(train_sample_new.vol_future).reshape(len(train_sample_new.vol_future), 1)
    # element wise multiplication of vol_today and predicted_y_train to get P
    P = np.array([a * b for a, b in zip(vol_today, predicted_y_train)])
    # implement matrix multiplication by using np.dot
    benchmark_delta_star = np.dot(np.transpose(P), (vol_tmr - vol_today)) / np.dot(np.transpose(P), P)

    return predicted_y_train, benchmark_delta_star.flatten()[0], fittedModel, train_sample_new

def SVM_Validation_Training(train_sample, forecaster, model, C, p=None, q=None):
    """
    :param train_sample: could be train_sample_daily or train_sample_weekly (obtained in main)
    :param forecaster: forecaster could take values 1, 2, 3, 4, 5 or 6
    :param model: "SVM","KernelSVM_rbf","KernelSVM_sigmoid"
    :param C: margin in the SVM model, needed to be optimized
    :param p: p is a parameter in forecaster 3 and forecaster 4 
    :param q: q is a parameter in forecaster 4    
    :return: MSE of one validation in the training sample where the entire training sample 
            is randomly divided into the training in sample set and the training out of sample set once
    """
    output = SVM_Predict_y_delta_star_training(train_sample, forecaster, model ,C, p, q)
    train_sample_new = output[3]
    benchmark_delta_star = output[1]

    index_train_in_sample = np.random.choice(len(train_sample_new), int(len(train_sample_new) * 4 / 5), replace=False)
    index_train_in_sample = np.sort(index_train_in_sample)
    index_train_out_of_sample = set(range(len(train_sample_new))) - set(index_train_in_sample)

    train_in_sample = train_sample_new.ix[index_train_in_sample]
    train_out_of_sample = train_sample_new.ix[index_train_out_of_sample]

    if model == "SVM":
        Model = SVC(C, kernel='linear')
    elif model == "KernelSVM_rbf":
        Model = SVC(C, kernel='rbf')
    elif model == "KernelSVM_sigmoid":
        Model = SVC(C, kernel='sigmoid')

    if forecaster == 1:
        fittedModel = Model.fit(np.array(train_in_sample.vol_now).reshape(len(train_in_sample.vol_now),1),
                                np.array(train_in_sample.label))
        predicted_y_train_out_of_sample =  fittedModel.predict(np.array(train_out_of_sample.vol_now).reshape(len(train_out_of_sample.vol_now),1))
    elif forecaster == 2:
        inputdf = train_in_sample[['vol_now', 'ret_now']]
        fittedModel = Model.fit(np.array(inputdf).reshape(len(inputdf),2),
                                np.array(train_in_sample.label))
        inputdf2 = train_out_of_sample[['vol_now', 'ret_now']]
        predicted_y_train_out_of_sample = fittedModel.predict(np.array(inputdf2).reshape(len(inputdf2),2))
    elif forecaster == 3:
        fittedModel = Model.fit(np.array(train_in_sample.E_p).reshape(len(train_in_sample.E_p), 1),
                                np.array(train_in_sample.label))
        predicted_y_train_out_of_sample = fittedModel.predict(np.array(train_out_of_sample.E_p).reshape(len(train_out_of_sample.E_p), 1))
    elif forecaster == 4:
        inputdf = train_in_sample[['E_p', 'E_q']]
        fittedModel = Model.fit(np.array(inputdf).reshape(len(inputdf), 2),
                                np.array(train_in_sample.label))
        inputdf2 = train_out_of_sample[['E_p', 'E_q']]
        predicted_y_train_out_of_sample = fittedModel.predict(np.array(inputdf2).reshape(len(inputdf2), 2))
    elif forecaster == 5:
        inputdf = train_in_sample[['vol_now','ret_now', 'vol_sqr_now']]
        fittedModel = Model.fit(np.array(inputdf).reshape(len(inputdf),3),
                                np.array(train_in_sample.label))
        inputdf2 = train_out_of_sample[['vol_now','ret_now', 'vol_sqr_now']]
        predicted_y_train_out_of_sample = fittedModel.predict(np.array(inputdf2).reshape(len(inputdf2),3))
    elif forecaster == 6:
        inputdf = train_in_sample[['vol_now',  'ret_now','volxret_now','vol_sqr_now']]
        fittedModel = Model.fit(np.array(inputdf).reshape(len(inputdf),4),
                                np.array(train_in_sample.label))
        inputdf2 = train_out_of_sample[['vol_now',  'ret_now','volxret_now','vol_sqr_now']]
        predicted_y_train_out_of_sample = fittedModel.predict(np.array(inputdf2).reshape(len(inputdf2),4))

    element1 = np.array(train_out_of_sample.vol_now)
    element2 = 1 + benchmark_delta_star * predicted_y_train_out_of_sample
    predicted_train_out_of_sample_vol_future = pd.Series([a * b for a, b in zip(element1, element2)])
    observed_train_out_of_sample_vol_future = pd.Series(train_out_of_sample.vol_future)

    # measuring prediction performance in the training our of sample
    Performance_ = PerformanceMeasure()
    MSE = Performance_.mean_se(observed_train_out_of_sample_vol_future,predicted_train_out_of_sample_vol_future)
    return MSE



def SVM_Optimize_Parameters(train_sample, forecaster, numCV, model, C_seq, time, name):
    """
    :param train_sample: could be train_sample_daily or train_sample_weekly (obtained in main)
    :param forecaster: forecaster could take values 1, 2, 5 or 6
    :param numCV: if numCV is 10, then 10-fold CV is conducted
    :param model: "SVM","KernelSVM_rbf","KernelSVM_sigmoid"
    :param C_seq: a sequence values for margin C in (Kernel) SVM models.
    :param time: time could be "Daily" or "Weekly"
    :param name: name of the currency pair
    :return: optimized parameters and give the plots of average MSE simulated against C (and p and q)
    """
    p_seq = [3, 5, 10]
    q_seq = [3, 5, 10]
    # C_seq = np.arange(0.5,5,0.5)

    mean_MSEs = []
    if forecaster == 1 or forecaster == 2 or forecaster == 5 or forecaster == 6:
        for n in range(len(C_seq)):
            MSEs = []
            for i in range(numCV):
                MSEs.append( SVM_Validation_Training(train_sample, forecaster, model, C =C_seq[n], p = None, q=None) )
            mean_MSEs.append(np.mean(MSEs))
        optimal_C = C_seq[mean_MSEs.index(min(mean_MSEs))]
        optimal_p = None
        optimal_q = None

        fig = plt.figure(figsize=(15, 10))
        ax = plt.plot(np.array(C_seq),np.array(mean_MSEs),'-o')
        plt.xlabel("margin C")
        plt.ylabel("Average MSE")
        title = name.replace(".csv", "") + " "+ time + " Average MSE of "+str(numCV)+"-fold CV for " + model + " Forecaster" + str(forecaster)
        plt.title(title+'\n Optimal C = ' + str(optimal_C))
        # plt.show()
        plt.savefig(title+'.png')

    if forecaster == 3:
        for n in range(len(C_seq)):
            for k in range(len(p_seq)):
                MSEs = []
                for i in range(numCV):
                    MSEs.append( SVM_Validation_Training(train_sample, forecaster, model, C =C_seq[n], p = p_seq[k], q=None) )
                mean_MSEs.append(np.mean(MSEs))
        min_index = mean_MSEs.index(min(mean_MSEs))
        optimal_C = C_seq[int(min_index/3)]
        optimal_p = p_seq[min_index % 3]
        optimal_q = None

        fig = plt.figure(figsize=(15, 10))
        ax = plt.axes(projection='3d')
        ax.scatter(np.array(np.repeat(C_seq,3)),np.array(p_seq*len(C_seq)),np.array(mean_MSEs), '-b')
        ax.set_xlabel("margin C")
        ax.set_ylabel("p")
        ax.set_zlabel("Average MSE")
        title = name.replace(".csv", "") + " "+ time + " Average MSE of "+str(numCV)+"-fold CV for " + model + " Forecaster" + str(forecaster)
        plt.title(title+'\nOptimal C = ' + str(optimal_C) + ' Optimal p = ' + str(optimal_p))
        # plt.show()
        plt.savefig(title+'.png')

    elif forecaster == 4:
        for n in range(len(C_seq)):
            for k in range(len(p_seq)):
                for m in range(len(q_seq)):
                    MSEs = []
                    for i in range(numCV):
                        MSEs.append( SVM_Validation_Training(train_sample, forecaster, model, C =C_seq[n], p = p_seq[k], q=q_seq[m]) )
                    mean_MSEs.append(np.mean(MSEs))
        min_index = mean_MSEs.index(min(mean_MSEs))
        optimal_C = C_seq[int(min_index/9)]
        optimal_p = p_seq[int(min_index/3) % 3]
        optimal_q = q_seq[min_index % 3]

        # 4D plot
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(45, 60)
        # hot: sequential black (corresponds to small value)-red-yellow-white (corresponds to small value), to emulate blackbody radiation from an object at increasing temperatures
        ax.scatter(np.array(np.repeat(C_seq,9)),np.array(np.repeat(p_seq,3).tolist()*len(C_seq)),np.array(q_seq*(3*len(C_seq))),c=np.array(mean_MSEs),cmap="hot")
        ax.set_xlabel("margin C")
        ax.set_ylabel("p")
        ax.set_zlabel("q")
        title = name.replace(".csv", "") + " "+ time + " Average MSE of "+str(numCV)+"-fold CV for " + model + " Forecaster" + str(forecaster)
        plt.title(title+'\n Optimal C = '+str(optimal_C) + ' Optimal p = '+str(optimal_p) + ' Optimal q = '+str(optimal_q))
        # plt.show()
        plt.savefig(title+'.png')


    return optimal_C, optimal_p, optimal_q, MSEs


def SVM_test_performance(train_sample, test_sample, forecaster,numCV,model, C_seq,time,name):
    """
    :param train_sample: could be train_sample_daily or train_sample_weekly (obtained in main)
    :param test_sample: could be test_sample_daily or test_sample_weekly (obtained in main)
    :param forecaster: forecaster could take values 1, 2, 5 or 6
    :param numCV: if numCV is 10, then 10-fold CV is conducted
    :param model: "SVM","KernelSVM_rbf","KernelSVM_sigmoid"
    :param C: margin in the SVM model, needed to be optimized
    :param time: time could be "Daily" or "Weekly"
    :param name: name of the currency pair
    :return: optimized parameters and prediction performance measure in the test sample
    """

    Optimized_Parameters = SVM_Optimize_Parameters(train_sample, forecaster, numCV, model, C_seq, time, name)
    C = Optimized_Parameters[0]
    p = Optimized_Parameters[1]
    q = Optimized_Parameters[2]

    test_sample_new  = SVM_Predict_y_delta_star_training(test_sample,forecaster,model, C,p,q)[3]
    training_output = SVM_Predict_y_delta_star_training(train_sample, forecaster,model, C, p, q)
    delta_tilta_star = training_output[1]
    fitted_model_training = training_output[2]

    if forecaster == 1:
        predicted_y_test = fitted_model_training.predict(np.array(test_sample_new.vol_now).reshape(len(test_sample_new.vol_now), 1))
    elif forecaster == 2:
        inputdf = test_sample_new[['vol_now', 'ret_now']]
        predicted_y_test = fitted_model_training.predict(np.array(inputdf).reshape(len(inputdf),2))
    elif forecaster == 3:
        predicted_y_test = fitted_model_training.predict(np.array(test_sample_new.E_p).reshape(len(test_sample_new.E_p), 1))
    elif forecaster == 4:
        inputdf = test_sample_new[['E_p', 'E_q']]
        predicted_y_test = fitted_model_training.predict(np.array(inputdf).reshape(len(inputdf), 2))
    elif forecaster == 5:
        inputdf = test_sample_new[['vol_now','ret_now', 'vol_sqr_now']]
        predicted_y_test = fitted_model_training.predict(np.array(inputdf).reshape(len(inputdf), 3))
    elif forecaster == 6:
        inputdf = test_sample_new[['vol_now',  'ret_now','volxret_now','vol_sqr_now']]
        predicted_y_test = fitted_model_training.predict(np.array(inputdf).reshape(len(inputdf), 4))

    element1 = np.array(test_sample_new.vol_now)
    element2 = 1 + delta_tilta_star * predicted_y_test
    predicted_test_sample_vol_future = pd.Series([a * b for a, b in zip(element1, element2)])
    observed_test_sample_vol_future = pd.Series(test_sample_new.vol_future)

    # measuring prediction performance in the test sample
    Performance_ = PerformanceMeasure()
    MSE = Performance_.mean_se(observed_test_sample_vol_future,predicted_test_sample_vol_future)
    QL = Performance_.quasi_likelihood(observed_test_sample_vol_future.astype('float64'),
                                       predicted_test_sample_vol_future.astype('float64'))
    SE = [(observed_test_sample_vol_future[i] - predicted_test_sample_vol_future[i])**2 for i in range(len(observed_test_sample_vol_future))]
    ln_SE = pd.Series(np.log(SE))
    return MSE, QL, ln_SE, C, p, q

