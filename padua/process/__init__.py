import pandas as pd
import numpy as np
import re
import itertools
from copy import copy

def numeric(s):
    """

    :param s:
    :return:
    """
    try:
        return int(s)
    except ValueError:
    
        try:
            return float(s)
        except ValueError:
            return s


def build_index_from_design(df, design, remove=None, types=None, axis=1, auto_convert_numeric=True, unmatched_columns='index'):
    """
    Build a MultiIndex from a design table.

    Supply with a table with column headings for the new multiindex
    and a index containing the labels to search for in the data.

    :param df:
    :param design:
    :param remove:
    :param types:
    :param axis:
    :param auto_convert_numeric:
    :return:
    """

    df = df.copy()
    if 'Label' not in design.index.names:
        design = design.set_index('Label')


    if remove is None:
        remove = []

    unmatched_for_index = []

    labels = design.index.values
    names = design.columns.values
    idx_levels = len(names)
    indexes = []
    
    # Convert numeric only columns_to_combine; except index
    if auto_convert_numeric:
        design = design.convert_objects(convert_numeric=True)
        # The match columns are always strings, so the index must also be
        design.index = design.index.astype(str)
    
    # Apply type settings
    if types:
        for n, t in types.items():
            if n in design.columns.values:
                design[n] = design[n].astype(t)
    
    # Build the index
    for lo in df.columns.values:
        l = copy(lo)
        for s in remove:
            l = l.replace(s, '')
        
        # Remove trailing/forward spaces
        l = l.strip()
        # Convert to numeric if possible
        l = numeric(l)
        # Attempt to match to the labels
        try:
            # Index
            idx = design.loc[str(l)]

        except:
            if unmatched_columns:
                unmatched_for_index.append(lo)
            else:
                # No match, fill with None
                idx = tuple([None] * idx_levels)
                indexes.append(idx)

        else:
            # We have a matched row, store it
            idx = tuple(idx.values)
            indexes.append(idx)

    if axis == 0:
        df.index = pd.MultiIndex.from_tuples(indexes, names=names)
    else:

        # If using unmatched for index, append
        if unmatched_columns == 'index':
            df = df.set_index(unmatched_for_index, append=True)

        elif unmatched_columns == 'drop':
            df = df.drop(unmatched_for_index, axis=1)

        df.columns = pd.MultiIndex.from_tuples(indexes, names=names)
    
    return df
    

def build_index_from_labels(df, indices, remove=None, types=None, axis=1):
    """
    Build a MultiIndex from a list of labels and matching regex

    Supply with a dictionary of Hierarchy levels and matching regex to
    extract this level from the sample label

    :param df:
    :param indices: Tuples of indices ('label','regex') matches
    :param strip: Strip these strings from labels before matching (e.g. headers)
    :param axis=1: Axis (1 = columns, 0 = rows)
    :return:
    """

    df = df.copy()

    if remove is None:
        remove = []

    if types is None:
        types = {}

    idx = [df.index, df.columns][axis]

    indexes = []

    for l in idx.get_level_values(0):

        for s in remove:
            l = l.replace(s, '')

        ixr = []
        for n, m in indices:
            m = re.search(m, l)
            if m:
                r = m.group(1)

                if n in types:
                    # Map this value to a new type
                    r = types[n](r)
            else:
                r = None

            ixr.append(r)
        indexes.append( tuple(ixr) )

    if axis == 0:
        df.index = pd.MultiIndex.from_tuples(indexes, names=[n for n, _ in indices])
    else:
        df.columns = pd.MultiIndex.from_tuples(indexes, names=[n for n, _ in indices])

    return df


def get_unique_indices(df, axis=1):
    """

    :param df:
    :param axis:
    :return:
    """
    return dict(zip(df.columns.names, dif.columns.levels))


def strip_index_labels(df, strip, axis=1):
    """

    :param df:
    :param strip:
    :param axis:
    :return:
    """

    df = df.copy()

    if axis == 0:
        df.columns = [c.replace(strip, '') for c in df.columns]

    elif axis == 1:
        df.columns = [c.replace(strip, '') for c in df.columns]

    return df


def combine_numeric_columns(df, columns_to_combine, method=np.mean, remove_combined=True):

    """
    Combine columns, calculating the specified value for N columns


    :param df: Pandas dataframe
    :param columns_to_combine: A list of tuples containing the column names to combine
    :param method: Function defining how to combine, e.g. numpy.mean, numpy.median
    :return:
    """

    df = df.copy()

    for ca, cb in columns_to_combine:
        df["%s_(x+y)/2_%s" % (ca, cb)] = (df[ca] + df[cb]) / 2

    if remove_combined:
        for ca, cb in columns_to_combine:
            df.drop([ca, cb], inplace=True, axis=1)

    return df




def apply_experimental_design(df, f, prefix='Intensity '):
    """
    Load the experimental design template from MaxQuant and use it to apply the label names to the data columns.

    :param df:
    :param f: File path for the experimental design template
    :param prefix:
    :return: dt
    """

    df = df.copy()

    edt = pd.read_csv(f, sep='\t', header=0)

    edt.set_index('Experiment', inplace=True)

    new_column_labels = []
    for l in df.columns.values:
        try:
            l = edt.loc[l.replace(prefix, '')]['Name']
        except (IndexError, KeyError):
            pass

        new_column_labels.append(l)

    df.columns = new_column_labels
    return df


def transform_numeric_columns(df, fn=np.log2, prefix='Intensity '):
    """
    Apply transformation to expression columns.

    Default is log2 transform to expression columns beginning with Intensity


    :param df:
    :param prefix: The column prefix for expression columns
    :return:
    """
    df = df.copy()

    mask = np.array([l.startswith(prefix) for l in df.columns.values])
    df.iloc[:, mask] = fn(df.iloc[:, mask])
    
    
    return df

    
def fold_columns_to_rows(df, levels_from=2):
    """
    Take a levels from the columns and fold down into the row index.
    This destroys the existing index; existing rows will appear as
    columns under the new column index

    :param df:
    :param levels_from: The level (inclusive) from which column index will be folded
    :return:
    """
    
    df = df.copy()
    df.reset_index(inplace=True, drop=True) # Wipe out the current index
    df = df.T
    
    # Build all index combinations

    a = [list( set( df.index.get_level_values(i) ) ) for i in range(0, levels_from)]
    combinations = list(itertools.product(*a))
    
    names = df.index.names[:levels_from]
    
    concats = []
    for c in combinations:
        
        try:
            dfcc = df.loc[c]

        except KeyError:
            continue

        else:
            # Silly pandas
            if len(dfcc.shape) == 1:
                continue
        
            dfcc.columns = pd.MultiIndex.from_tuples([c]*dfcc.shape[1], names=names)
            concats.append(dfcc)

    # Concatenate
    dfc = pd.concat(concats, axis=1)
    dfc.sort_index(axis=1, inplace=True)

    # Fix name if collapsed
    if dfc.index.name is None:
        dfc.index.name = df.index.names[-1]

    return dfc
