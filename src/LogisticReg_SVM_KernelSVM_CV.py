import numpy as np
from sklearn.linear_model import LogisticRegression as Logit
from Performance_Measure import *
from SEplot import se_plot as SE
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC as SVC

"""
this Python profile implements 
(1) Logistic Regression for Forecaster 3 and 4
(2) SVM for Forecaster 1-6
(3) Kernel SVM for Forecaster 1-6

"""