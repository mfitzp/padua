import pandas as pd
import numpy as np
import requests

try:
    import sklearn
except ImportError:
    sklearn = False
else:
    from sklearn.decomposition import PCA    



try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from . import filters, process
from .utils import get_protein_id


def correlation(df, rowvar=False):
    """
    Calculate column-wise Pearson correlations using ``numpy.ma.corrcoef``

    Input data is masked to ignore NaNs when calculating correlations. Data is returned as
    a Pandas ``DataFrame`` of column_n x column_n dimensions, with column index copied to
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

    For more information on PCA and the algorithm used, see the `scikit-learn documentation <http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_.

    :param df: Pandas ``DataFrame`` to perform the analysis on
    :param n_components: ``int`` number of components to select
    :param mean_center: ``bool`` mean center the data before performing PCA
    :param kwargs: additional keyword arguments to `sklearn.decomposition.PCA`
    :return: scores ``DataFrame`` of PCA scores n_components x n_samples
             weights ``DataFrame`` of PCA weights n_variables x n_components
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
    scores.index = ['Principal Component %d (%.2f%%)' % ( (n+1), pca.explained_variance_ratio_[n]*100 ) for n in range(0, scores.shape[0])]
    scores.columns = df.columns

    weights = pd.DataFrame(pca.components_).T
    weights.index = df.index
    weights.columns = ['Weights on Principal Component %d' % (n+1) for n in range(0, weights.shape[1])]
       
    return scores, weights


def plsda(df, a, b, n_components=2, mean_center=False, scale=True, **kwargs):
    """
    Partial Least Squares Discriminant Analysis, based on `sklearn.cross_decomposition.PLSRegression`

    Performs a binary group partial least squares discriminant analysis (PLS-DA) on the supplied
    dataframe, selecting the first ``n_components``.

    Sample groups are defined by the selectors ``a`` and ``b`` which are used to select columns
    from the supplied dataframe. The result model is applied to the entire dataset,
    projecting non-selected samples into the same space.

    For more information on PLS regression and the algorithm used, see the `scikit-learn documentation <http://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html>`_.

    :param df: Pandas ``DataFrame`` to perform the analysis on
    :param a: Column selector for group a
    :param b: Column selector for group b
    :param n_components: ``int`` number of components to select
    :param mean_center: ``bool`` mean center the data before performing PLS regression
    :param kwargs: additional keyword arguments to `sklearn.cross_decomposition.PLSRegression`
    :return: scores ``DataFrame`` of PLSDA scores n_components x n_samples
             weights ``DataFrame`` of PLSDA weights n_variables x n_components
    """

    if not sklearn:
        assert('This library depends on scikit-learn (sklearn) to perform PLS-DA')

    from sklearn.cross_decomposition import PLSRegression

    df = df.copy()

    # We have to zero fill, nan errors in PLSRegression
    df[ np.isnan(df) ] = 0

    if mean_center:
        mean = np.mean(df.values, axis=0)
        df = df - mean

    sxa, _ = df.columns.get_loc_level(a)
    sxb, _ = df.columns.get_loc_level(b)

    dfa = df.iloc[:, sxa]
    dfb = df.iloc[:, sxb]

    dff = pd.concat([dfa, dfb], axis=1)
    y = np.ones(dff.shape[1])
    y[np.arange(dfa.shape[1])] = 0

    plsr = PLSRegression(n_components=n_components, scale=scale, **kwargs)
    plsr.fit(dff.values.T, y)

    # Apply the generated model to the original data
    x_scores = plsr.transform(df.values.T)

    scores = pd.DataFrame(x_scores.T)
    scores.index = ['Latent Variable %d' % (n+1) for n in range(0, scores.shape[0])]
    scores.columns = df.columns

    weights = pd.DataFrame(plsr.x_weights_)
    weights.index = df.index
    weights.columns = ['Weights on Latent Variable %d' % (n+1) for n in range(0, weights.shape[1])]

    loadings = pd.DataFrame(plsr.x_loadings_)
    loadings.index = df.index
    loadings.columns = ['Loadings on Latent Variable %d' % (n+1) for n in range(0, loadings.shape[1])]

    return scores, weights, loadings


def plsr(df, v, n_components=2, mean_center=False, scale=True, **kwargs):
    """
    Partial Least Squares Regression Analysis, based on `sklearn.cross_decomposition.PLSRegression`

    Performs a partial least squares regression (PLS-R) on the supplied dataframe ``df``
    against the provided continuous variable ``v``, selecting the first ``n_components``.

    For more information on PLS regression and the algorithm used, see the `scikit-learn documentation <http://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html>`_.

    :param df: Pandas ``DataFrame`` to perform the analysis on
    :param v: Continuous variable to perform regression against
    :param n_components: ``int`` number of components to select
    :param mean_center: ``bool`` mean center the data before performing PLS regression
    :param kwargs: additional keyword arguments to `sklearn.cross_decomposition.PLSRegression`
    :return: scores ``DataFrame`` of PLS-R scores n_components x n_samples
             weights ``DataFrame`` of PLS-R weights n_variables x n_components
    """

    if not sklearn:
        assert('This library depends on scikit-learn (sklearn) to perform PLS-DA')

    from sklearn.cross_decomposition import PLSRegression

    df = df.copy()

    # We have to zero fill, nan errors in PLSRegression
    df[ np.isnan(df) ] = 0

    if mean_center:
        mean = np.mean(df.values, axis=0)
        df = df - mean

    #TODO: Extract values if v is DataFrame?

    plsr = PLSRegression(n_components=n_components, scale=scale, **kwargs)
    plsr.fit(df.values.T, v)

    scores = pd.DataFrame(plsr.x_scores_.T)
    scores.index = ['Latent Variable %d' % (n+1) for n in range(0, scores.shape[0])]
    scores.columns = df.columns

    weights = pd.DataFrame(plsr.x_weights_)
    weights.index = df.index
    weights.columns = ['Weights on Latent Variable %d' % (n+1) for n in range(0, weights.shape[1])]

    loadings = pd.DataFrame(plsr.x_loadings_)
    loadings.index = df.index
    loadings.columns = ['Loadings on Latent Variable %d' % (n+1) for n in range(0, loadings.shape[1])]

    #r = plsr.score(df.values.T, v)
    predicted = plsr.predict(df.values.T)

    return scores, weights, loadings, predicted




def _non_zero_sum(df):
    # Following is just to build the template; actual calculate below
    dfo = df.sum(axis=0, level=0)

    for c in df.columns.values:
        dft = df[c]
        dfo[c] = dft[ dft > 0].sum(axis=0, level=0)

    return dfo


def enrichment_from_evidence(dfe, modification="Phospho (STY)"):
    """
    Calculate relative enrichment of peptide modifications from evidence.txt.

    Taking a modifiedsitepeptides ``DataFrame`` returns the relative enrichment of the specified
    modification in the table.

    The returned data columns are generated from the input data columns.

    :param df: Pandas ``DataFrame`` of evidence
    :return: Pandas ``DataFrame`` of percentage modifications in the supplied data.
    """

    dfe = dfe.reset_index().set_index('Experiment')

    dfe['Modifications'] = np.array([modification in m for m in dfe['Modifications']])
    dfe = dfe.set_index('Modifications', append=True)

    dfes = dfe.sum(axis=0, level=[0,1]).T

    columns = dfes.sum(axis=1, level=0).columns

    total = dfes.sum(axis=1, level=0).values.flatten() # Total values
    modified = dfes.iloc[0, dfes.columns.get_level_values('Modifications').values ].values # Modified
    enrichment = modified / total

    return pd.DataFrame([enrichment], columns=columns, index=['% Enrichment'])




def enrichment_from_msp(dfmsp, modification="Phospho (STY)"):
    """
    Calculate relative enrichment of peptide modifications from modificationSpecificPeptides.txt.

    Taking a modifiedsitepeptides ``DataFrame`` returns the relative enrichment of the specified
    modification in the table.

    The returned data columns are generated from the input data columns.

    :param df: Pandas ``DataFrame`` of modificationSpecificPeptides
    :return: Pandas ``DataFrame`` of percentage modifications in the supplied data.
    """

    dfmsp['Modifications'] = np.array([modification in m for m in dfmsp['Modifications']])
    dfmsp = dfmsp.set_index(['Modifications'])
    dfmsp = dfmsp.filter(regex='Intensity ')

    dfmsp[ dfmsp == 0] = np.nan
    df_r = dfmsp.sum(axis=0, level=0)

    modified = df_r.loc[True].values
    total = df_r.sum(axis=0).values
    enrichment = modified / total

    return pd.DataFrame([enrichment], columns=dfmsp.columns, index=['% Enrichment'])



def sitespeptidesproteins(df, site_localization_probability=0.75):
    """
    Generate summary count of modified sites, peptides and proteins in a processed dataset ``DataFrame``.

    Returns the number of sites, peptides and proteins as calculated as follows:

    - `sites` (>0.75; or specified site localization probability) count of all sites > threshold
    - `peptides` the set of `Sequence windows` in the dataset (unique peptides)
    - `proteins` the set of unique leading proteins in the dataset

    :param df: Pandas ``DataFrame`` of processed data
    :param site_localization_probability: ``float`` site localization probability threshold (for sites calculation)
    :return: ``tuple`` of ``int``, containing sites, peptides, proteins
    """

    sites = filters.filter_localization_probability(df, site_localization_probability)['Sequence window']
    peptides = set(df['Sequence window'])
    proteins = set([str(p).split(';')[0] for p in df['Proteins']])
    return len(sites), len(peptides), len(proteins)


def modifiedaminoacids(df):
    """
    Calculate the number of modified amino acids in supplied ``DataFrame``.

    Returns the total of all modifications and the total for each amino acid individually, as an ``int`` and a
    ``dict`` of ``int``, keyed by amino acid, respectively.

    :param df: Pandas ``DataFrame`` containing processed data.
    :return: total_aas ``int`` the total number of all modified amino acids
             quants ``dict`` of ``int`` keyed by amino acid, giving individual counts for each aa.
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
    Calculate gene ontology (GO) enrichment for a specified set of indices, using the PantherDB GO enrichment service.

    Provided with a processed data ``DataFrame`` will calculate the GO ontology enrichment specified by `enrichment`,
    for the specified `organism`. The IDs to use for genes are taken from the field `ids_from`, which by default is
    compatible with both proteinGroups and modified peptide tables. Setting the `fdr` parameter (default=0.05) sets
    the cut-off to use for filtering the results. If `summary` is ``True`` (default) the returned ``DataFrame``
    contains just the ontology summary and FDR.

    :param df: Pandas ``DataFrame`` to
    :param enrichment: ``str`` GO enrichment method to use (one of 'function', 'process', 'cellular_location', 'protein_class', 'pathway')
    :param organism: ``str`` organism name (e.g. "Homo sapiens")
    :param summary: ``bool`` return full, or summarised dataset
    :param fdr: ``float`` FDR cut-off to use for returned GO enrichments
    :param ids_from: ``list`` of ``str`` containing the index levels to select IDs from (genes, protein IDs, etc.) default=['Proteins','Protein IDs']
    :return: Pandas ``DataFrame`` containing enrichments, sorted by P value.
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

    return go.sort_values(by="P", ascending=True)
