from . import remove_columns_containing

def remove_reverse(df):
    """
    Remove rows with a + in the 'Reverse' column.

    Return a ``DataFrame`` where rows where there is a "+" in the column 'Reverse' are removed.
    Filters data to remove peptides matched as reverse.

    :param df: Pandas ``DataFrame``
    :return: filtered Pandas ``DataFrame``
    """
    return remove_columns_containing(df, 'Reverse', '+')

def remove_contaminants(df):
    """
    Remove rows with a + in the 'Contaminants' column

    Return a ``DataFrame`` where rows where there is a "+" in the column 'Contaminants' are removed.
    Filters data to remove peptides matched as reverse.

    :param df: Pandas ``DataFrame``
    :return: filtered Pandas ``DataFrame``
    """
    return remove_columns_containing(df, 'Contaminant', '+')

def remove_potential_contaminants(df):
    """
    Remove rows with a + in the 'Potential contaminant' column

    Return a ``DataFrame`` where rows where there is a "+" in the column 'Contaminants' are removed.
    Filters data to remove peptides matched as reverse.

    :param df: Pandas ``DataFrame``
    :return: filtered Pandas ``DataFrame``
    """
    return remove_columns_containing(df, 'Potential contaminant', '+')


def remove_only_identified_by_site(df):
    """
    Remove rows with a + in the 'Only identified by site' column

    Return a ``DataFrame`` where rows where there is a "+" in the column 'Only identified by site' are removed.
    Filters data to remove peptides matched as reverse.

    :param df: Pandas ``DataFrame``
    :return: filtered Pandas ``DataFrame``
    """
    return remove_columns_containing(df, 'Only identified by site', '+')


def filter_localization_probability(df, threshold=0.75):
    """
    Remove rows with a localization probability below 0.75

    Return a ``DataFrame`` where the rows with a value < `threshold` (default 0.75) in column 'Localization prob' are removed.
    Filters data to remove poorly localized peptides (non Class-I by default).

    :param df: Pandas ``DataFrame``
    :param threshold: Cut-off below which rows are discarded (default 0.75)
    :return: Pandas ``DataFrame``
    """
    df = df.copy()
    localization_probability_mask = df['Localization prob'].values >= threshold
    return df.iloc[localization_probability_mask, :]
