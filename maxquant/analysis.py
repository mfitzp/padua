__author__ = 'Fitzp002'
import pandas as pd
import numpy as np


try:
    import sklearn
except:
    sklearn = False
else:
    from sklearn.decomposition import PCA    


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
    
    
def pca(df, n_components=2, mean_center=False, *args, **kwargs):
    if not sklearn:
        assert('This library depends on scikit-learn (sklearn) to perform PCA analysis')
        
    from sklearn.decomposition import PCA

    df = df.copy()
    
    # We have to zero fill, nan errors in PCA
    df[ np.isnan(df) ] = 0

    if mean_center:
        mean = np.mean(df.values, axis=0)
        df = df - mean

    pca = PCA(n_components=n_components, *args, **kwargs)
    pca.fit(df.values.T)

    scores = pd.DataFrame(pca.transform(df.values.T)).T
    scores.index =  ['Principal Component %d' % (n+1) for n in range(0, scores.shape[0])]
    scores.columns = df.columns

    weights = pd.DataFrame(pca.components_).T
    weights.index = df.index
    weights.columns =  ['Weights on Principal Component %d' % (n+1) for n in range(0, weights.shape[1])]
       
    return scores, weights
