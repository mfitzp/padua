__author__ = 'Fitzp002'
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


def get_protein_id(s):
    return str(s).split(';')[0].split(' ')[0].split('_')[0]


def correlation(df):
    """
    Calculate column-wise Pearson correlations



    :param df:
    :return:
    """

    # Create a correlation matrix for all correlations
    # of the columns (filled with na for all values)
    df = df.copy()
    df[ np.isinf(df) ] = np.nan

    n = df.shape[1]

    cdf = np.zeros((n, n))
    cdf[ cdf == 0] = np.nan

    last = -1

    for y in range(n):
        for x in range(y, n):
            if x == y:
                r = 1
            else:
                data = df.ix[:, [x, y] ].dropna(how='any').values
                r = np.corrcoef(data[:, 0], data[:, 1])[0, 1]

            cdf[x, y] = r ** 2
            cdf[y, x] = r ** 2

    cdf = pd.DataFrame(cdf)
    cdf.columns = df.columns
    cdf.index = df.columns

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
    
    
def enrichment(df):

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
    sites = filters.filter_localization_probability(df, site_localization_probability)['Sequence window']
    peptides = set(df['Sequence window'])
    proteins = set([p.split(';')[0] for p in df['Proteins']])
    return len(sites), len(peptides), len(proteins)
    

def modifiedaminoacids(df):
    amino_acids = list(df['Amino acid'].values)
    aas = set(amino_acids)
    quants = {}

    for aa in aas:
        quants[aa] = amino_acids.count(aa)
        
    total_aas = len(amino_acids)

    return total_aas, quants


def go_enrichment(l, enrichment='function', summary=True, fdr=0.05):

    data = "\n".join([get_protein_id(s) for s in l])
    r = requests.post("http://www.pantherdb.org/webservices/garuda/enrichment.jsp", data={
            'organism':"Homo sapiens",
            'type':'enrichment',
            'enrichmentType': enrichment},
            files = {
            'geneList': ('genelist.txt', StringIO(data) ),

            }
        )

    go = pd.read_csv(StringIO(r.text), sep='\t', skiprows=5, lineterminator='\n', header=None)
    go.columns = ["GO", "Name", "Protein", "P"]
    go = go.set_index(["GO","Name"])
    if summary:
        go = go.drop("Protein", axis=1).mean(axis=0, level=["GO","Name"])

    return go[ go["P"] < fdr ].sort("P", ascending=True)
