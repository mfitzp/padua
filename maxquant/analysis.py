__author__ = 'Fitzp002'
import pandas as pd
import numpy as np

def column_correlations(df):
    """
    Calculate column-wise Pearson correlations



    :param df:
    :return:
    """

    # Create a correlation matrix for all correlations
    # of the columns (filled with na for all values)
    df = df.copy()
    df[ np.isinf(df) ] = np.nan

    n = len( df.columns.get_level_values(0) )


    data = np.zeros((n, n))
    data[data == 0] = np.nan

    cdf = pd.DataFrame(data)
    cdf.columns = df.columns
    cdf.index = df.columns

    for y in range(n):
        for x in range(y, n):
            try:
                data = df.iloc[:, [x, y] ].dropna(how='any').values
                r = np.corrcoef(data[:, 0], data[:, 1])[0, 1]

            except TypeError:
                pass

            else:
                cdf.iloc[x, y] = r **2
                cdf.iloc[y, x] = r **2

    return cdf