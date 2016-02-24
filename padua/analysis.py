import pandas as pd
import numpy as np
import requests

try:
    import sklearn
except:
    sklearn = False
else:
    from sklearn.decomposition import PCA    

from . import filters

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


from .utils import get_protein_id


def correlation(df, rowvar=False):
    """
    Calculate column-wise Pearson correlations using ``numpy.ma.corrcoef``

    Input data is masked to ignore NaNs when calculating correlations. Data is returned as
    a Pandas ``DataFrame` of column_n x column_n dimensions, with column index copied to
    both axes.

    :param df: Pandas DataFrame
    :return: Pandas DataFrame (n_columns x n_columns) of column-wise correlations
    """

    # Create a correlation matrix for all correlations
    # of the columns (filled with na for all values)
    df = df.copy()
    maskv = np.ma.masked_where(np.isnan(df.values), df.values)
    cdf = np.ma.corrcoef(maskv, rowvar=False)
    cdf = pd.DataFrame(np.array(cdf))
    cdf.columns = df.columns
    cdf.index = df.columns

    return cdf
    
    
def pca(df, n_components=2, mean_center=False, **kwargs):
    """
    Principal Component Analysis, based on `sklearn.decomposition.PCA`

    Performs a principal component analysis (PCA) on the supplied dataframe, selecting the first ``n_components`` components
    in the resulting model. The model scores and weights are returned.

    For more information on PCA and the algorithm used, see the `scikit-learn documentation <http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`.

    :param df: Pandas ``DataFrame`` to perform the analysis on
    :param n_components: ``int`` number of components to select
    :param mean_center: ``bool`` mean center the data before performing PCA
    :param kwargs: additional keyword arguments to `sklearn.decomposition.PCA`
    :return: scores ``DataFrame`` of PCA scores n_components x n_samples
             weights ``DataFrame`` of PCA scores n_variables x n_components
    """

    if not sklearn:
        assert('This library depends on scikit-learn (sklearn) to perform PCA analysis')
        
    from sklearn.decomposition import PCA

    df = df.copy()
    
    # We have to zero fill, nan errors in PCA
    df[ np.isnan(df) ] = 0

    if mean_center:
        mean = np.mean(df.values, axis=0)
        df = df - mean

    pca = PCA(n_components=n_components, **kwargs)
    pca.fit(df.values.T)

    scores = pd.DataFrame(pca.transform(df.values.T)).T
    scores.index =  ['Principal Component %d' % (n+1) for n in range(0, scores.shape[0])]
    scores.columns = df.columns

    weights = pd.DataFrame(pca.components_).T
    weights.index = df.index
    weights.columns =  ['Weights on Principal Component %d' % (n+1) for n in range(0, weights.shape[1])]
       
    return scores, weights



def enrichment(df):
    """
    Calculate relative enrichment of peptide modifications.


    :param df: Pandas ``DataFrame``
    :return:
    """

    values = []
    groups = []
    #totals = []

    dfr = df.sum(axis=1, level=0)
    for c in dfr.columns.values:
        dfx = dfr[c]
        dfx = dfx.dropna(axis=0, how='any')
        #total = len([m for m in dfx.index.values if m != 'Unmodified'])
        total = dfx.index.values.shape[0]
        # Sum up the number of phosphosites
        dfx = dfx.reset_index().filter(regex='Sequence|Modifications').set_index('Sequence').sum(axis=0, level=0)
        phosp = dfx[dfx > 0].shape[0]

        values.append((phosp, total-phosp))
        groups.append(c)
        #totals.append(total)

    return pd.DataFrame(np.array(values).T, columns=groups, index=["",""])
    

def sitespeptidesproteins(df, site_localization_probability=0.75):
    """


    :param df:
    :param site_localization_probability:
    :return:
    """
    sites = filters.filter_localization_probability(df, site_localization_probability)['Sequence window']
    peptides = set(df['Sequence window'])
    proteins = set([p.split(';')[0] for p in df['Proteins']])
    return len(sites), len(peptides), len(proteins)


def modifiedaminoacids(df):
    """


    :param df:
    :return:
    """
    amino_acids = list(df['Amino acid'].values)
    aas = set(amino_acids)
    quants = {}

    for aa in aas:
        quants[aa] = amino_acids.count(aa)
        
    total_aas = len(amino_acids)

    return total_aas, quants


def go_enrichment(df, enrichment='function', organism='Homo sapiens', summary=True, fdr=0.05, ids_from=['Proteins','Protein IDs']):
    """
    
    :param df:
    :param enrichment:
    :param organism:
    :param summary:
    :param fdr:
    :param ids_from:
    :return:
    """
    if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
        l = list(set(ids_from) & set(df.index.names))[0]
        data = "\n".join([get_protein_id(s) for s in df.index.get_level_values(l)])
    else:
        data = "\n".join([get_protein_id(s) for s in df])

    r = requests.post("http://www.pantherdb.org/webservices/garuda/enrichment.jsp", data={
            'organism': organism,
            'type': 'enrichment',
            'enrichmentType': enrichment},
            files = {
            'geneList': ('genelist.txt', StringIO(data) ),

            }
        )

    try:
        go = pd.read_csv(StringIO(r.text), sep='\t', skiprows=5, lineterminator='\n', header=None)
    except ValueError:
        return None

    go.columns = ["GO", "Name", "Protein", "P"]
    go = go.set_index(["GO","Name"])
    if summary:
        go = go.drop("Protein", axis=1).mean(axis=0, level=["GO","Name"])

    if fdr:
        go = go[ go["P"] < fdr ]

    return go.sort("P", ascending=True)
