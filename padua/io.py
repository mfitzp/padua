import pandas as pd
import numpy as np

from .utils import get_protein_id

def read_maxquant(f, header=0, index_col='id', **kwargs):
    """
    Load the quantified table output from MaxQuant run, e.g.

        - Proteingroups.txt
        - Phospho (STY)Sites.txt

    :param f: Source file
    :return: Pandas dataframe of imported data
    """
    df = pd.read_csv(f, delimiter='\t', header=header, index_col=index_col, **kwargs)

    return df


def read_perseus(f):
    """
    Load a Perseus processed data table

    :param f: Source file
    :return: Pandas dataframe of imported data
    """
    df = pd.read_csv(f, delimiter='\t', header=[0,1,2,3], low_memory=False)
    df.columns = pd.MultiIndex.from_tuples([(x,) for x in df.columns.get_level_values(0)])
    return df


def write_perseus(f, df):
    """
    Export a dataframe to Perseus; recreating the format

    :param f:
    :param df:
    :return:
    """

    ### Generate the Perseus like type index

    FIELD_TYPE_MAP = {
        'Amino acid':'C',
        'Charge':'C',
        'Reverse':'C',
        'Potential contaminant':'C',
        'Multiplicity':'C',
        'Localization prob':'N',
        'PEP':'N',
        'Score':'N',
        'Delta score':'N',
        'Score for localization':'N',
        'Mass error [ppm]':'N',
        'Intensity':'N',
        'Position':'N',
        'Proteins':'T',
        'Positions within proteins':'T',
        'Leading proteins':'T',
        'Protein names':'T',
        'Gene names':'T',
        'Sequence window':'T',
        'Unique identifier':'T',
    }

    def map_field_type(n, c):
        try:
            t = FIELD_TYPE_MAP[c]
        except:
            t = "E"

        # In the first element, add type indicator
        if n == 0:
            t = "#!{Type}%s" % t

        return t

    df = df.copy()
    df.columns = pd.MultiIndex.from_tuples([(k, map_field_type(n, k)) for n, k in enumerate(df.columns)], names=["Label","Type"])
    df = df.transpose().reset_index().transpose()
    df.to_csv(f, index=False, header=False)
    

def write_phosphopath(df, f, extra_columns=None):
    """
    Write out the data frame of phosphosites in the following format::

        protein, protein-Rsite, Rsite, multiplicity
        Q13619	Q13619-S10	S10	1
        Q9H3Z4	Q9H3Z4-S10	S10	1
        Q6GQQ9	Q6GQQ9-S100	S100	1
        Q86YP4	Q86YP4-S100	S100	1
        Q9H307	Q9H307-S100	S100	1
        Q8NEY1	Q8NEY1-S1000	S1000	1

    The file is written as a comma-separated (CSV) file to file ``f``.

    :param df:
    :param f:
    :return:
    """

    def _protein_id(s): return str(s).split(';')[0].split(' ')[0].split('_')[0].split('-')[0]

    proteins = [_protein_id(k) for k in df.index.get_level_values('Proteins')]
    amino_acids = df.index.get_level_values('Amino acid')
    positions = [_protein_id(k) for k in df.index.get_level_values('Positions within proteins')]
    multiplicity = [k[-1] for k in df.index.get_level_values('Multiplicity')]

    apos = ["%s%s" % x for x in zip(amino_acids, positions)]
    prar = ["%s-%s" % x for x in zip(proteins, apos)]

    phdf = pd.DataFrame(np.array(list(zip(proteins, prar, apos, multiplicity))))
    if extra_columns:
        for c in extra_columns:
            phdf[c] = df[c].values

    phdf.to_csv(f, sep='\t', index=None, header=None)

def write_phosphopath_ratio(df, f, v, a=None, b=None):
    """
    Write out the data frame ratio between two groups
    protein-Rsite-multiplicity-timepoint
    ID	Ratio
    Q13619-S10-1-1	0.5
    Q9H3Z4-S10-1-1	0.502
    Q6GQQ9-S100-1-1	0.504
    Q86YP4-S100-1-1	0.506
    Q9H307-S100-1-1	0.508
    Q8NEY1-S1000-1-1	0.51
    Q13541-S101-1-1	0.512
    O95785-S1012-2-1	0.514
    O95785-S1017-2-1	0.516
    Q9Y4G8-S1022-1-1	0.518
    P35658-S1023-1-1	0.52

    :param df:
    :param f:
    :param v: Value ratio
    :param t: Timepoint
    :param a:
    :param b:
    :return:
    """

    proteins = [get_protein_id(k) for k in df.index.get_level_values('Proteins')]
    amino_acids = df.index.get_level_values('Amino acid')
    positions = [get_protein_id(k) for k in df.index.get_level_values('Positions within proteins')]
    multiplicity = [int(k[-1]) for k in df.index.get_level_values('Multiplicity')]

    apos = ["%s%s" % x for x in zip(amino_acids, positions)]

    prar = ["%s-%s-%d-1" % x for x in zip(proteins, apos, multiplicity)]

    phdf = pd.DataFrame(np.array(list(zip(prar, v))))
    phdf.columns = ["ID", "Ratio"]
    phdf.to_csv(f, sep='\t', index=None)


def write_r(df, f, sep=",", index_join="@", columns_join="."):
    """
    Export dataframe in a format easily importable to R

    Index fields are joined with "@" and column fields by "." by default.
    :param df:
    :param f:
    :param index_join:
    :param columns_join:
    :return:
    """

    df = df.copy()
    df.index = ["@".join([str(s) for s in v]) for v in df.index.values]
    df.columns = [".".join([str(s) for s in v]) for v in df.index.values]
    df.to_csv(f, sep=sep)


