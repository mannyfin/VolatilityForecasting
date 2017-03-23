import pandas as pd
import numpy as np


def result_to_df_list(list_name=list, method_result=tuple, index_value=list, column_value=list):
    """

    :param list_name: =list(). This should be initialized outside of this function. Feel free to pass empty list
    :param method_result: This is the tuple result of the method ex. (MSE,QL)
    :param index_value: This is the method name, passed as an array. Can be an array of strings too.
    :param column_value: This is the column values. the len of column values should be the same as the tuple
    :return:
    """

    # has issues if it is passed a dataframe. May need to create a Multi-level dataframe
    # reshape this
    method_result = np.reshape(method_result, [len(index_value), len(column_value)])
    df = pd.DataFrame()
    df = df.append(pd.DataFrame(method_result, index=index_value, columns=column_value))
    list_name.append(df)
    return list_name


def list_to_df(list_name=list):
    """
    Quick way to concatenate the list of df's into a single df
    :param list_name:
    :return:
    """
    df = pd.DataFrame()
    df = df.append(list_name)
    return df