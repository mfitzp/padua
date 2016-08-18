import pandas as pd
import numpy as np

from ..utils import get_protein_id


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
