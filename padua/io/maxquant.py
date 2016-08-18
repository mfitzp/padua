import pandas as pd


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
