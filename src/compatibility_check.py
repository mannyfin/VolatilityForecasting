import pandas as pd


def compatibility_check(observed=None, prediction=None):
    if isinstance(observed, pd.core.frame.DataFrame) is True:
        prediction = pd.core.frame.DataFrame(prediction)

    # shouldnt have this issue below, but putting this here just in case
    elif isinstance(observed, pd.core.series.Series) is True:
        prediction = pd.core.series.Series(prediction)

    return prediction
