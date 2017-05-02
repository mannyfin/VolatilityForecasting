import pandas as pd
# TODO drop unnamed 0 col option if desir
def split_data(dataframe=None, idx=1, reset_index=False):
    train = dataframe[:idx]
    test = dataframe[idx:]

    if reset_index is True:
        train = train.reset_index()
        test = test.reset_index()

    return train, test