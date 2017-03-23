from VAR2 import *
import numpy as np


def lassofn(y, xw, alpha = 0.1, stringin):
    q = 9
    p = 3
    n = p * 100
    # lasso: do mse and ql
