import pandas as pd
import os


def prediction_output(prediction, method_number):
    assert isinstance(prediction, pd.core.frame.DataFrame), "is this a df?"
    prediction.to_csv(os.getcwd() + '\\test_prediction_'+str(method_number)+'.csv')


