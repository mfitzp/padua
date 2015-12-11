import numpy as np


def subtract_column_median(df, prefix='Intensity '):
    """
    Apply column-wise normalisation to expression columns.

    Default is median transform to expression columns beginning with Intensity


    :param df:
    :param prefix: The column prefix for expression columns
    :return:
    """
    df = df.copy()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    mask = [l.startswith(prefix) for l in df.columns.values]
    df.iloc[:, mask] = df.iloc[:, mask] - df.iloc[:, mask].median(axis=0)

    return df
